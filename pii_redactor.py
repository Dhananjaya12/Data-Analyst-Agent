"""
Production-Grade PII Redactor with Encryption
"""

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from cryptography.fernet import Fernet
from presidio_analyzer.nlp_engine import NlpEngineProvider
import sqlite3
import pandas as pd
import hashlib
import os
from datetime import datetime, timedelta
from typing import Optional
import json


class SecurePIIRedactor:
    """
    Production-grade PII redactor with:
    - Encrypted storage
    - Per-user access control
    - Audit logging
    - Auto-cleanup
    """
    
    def __init__(self, db_path='secure_pii.db', encryption_key_path='encryption.key'):
        # self.analyzer = AnalyzerEngine()
        # Configure to use small model (en_core_web_sm)
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
        }

        provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
        nlp_engine = provider.create_engine()

        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine)

        self.anonymizer = AnonymizerEngine()
        
        # Encryption setup
        self.encryption_key = self._load_or_create_key(encryption_key_path)
        self.cipher = Fernet(self.encryption_key)
        
        # Database setup
        self.db_path = db_path
        self._init_database()
        
        print("🔒 Secure PII Redactor initialized")
    
    def _load_or_create_key(self, key_path):
        """Load existing key or create new one"""
        if os.path.exists(key_path):
            with open(key_path, 'rb') as f:
                key = f.read()
            print(f"🔑 Loaded encryption key from {key_path}")
        else:
            key = Fernet.generate_key()
            with open(key_path, 'wb') as f:
                f.write(key)
            print(f"🔑 Created new encryption key: {key_path}")
            print("⚠️  BACKUP THIS KEY - data cannot be recovered without it!")
        
        return key
    
    def _init_database(self):
        """Initialize SQLite database with tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Mappings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pii_mappings (
                token TEXT PRIMARY KEY,
                encrypted_value BLOB NOT NULL,
                entity_type TEXT NOT NULL,
                user_id TEXT NOT NULL,
                file_id TEXT NOT NULL,
                column_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                accessed_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP
            )
        ''')
        
        # Audit log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                token TEXT,
                user_id TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                details TEXT
            )
        ''')
        
        # Indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON pii_mappings(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_file_id ON pii_mappings(file_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON pii_mappings(created_at)')
        
        conn.commit()
        conn.close()
        
        print(f"📊 Database initialized: {self.db_path}")
    
    def redact_dataframe(self, df: pd.DataFrame, file_id: str, user_id: str) -> pd.DataFrame:
        """
        Redact PII from entire dataframe
        
        Args:
            df: Pandas dataframe
            file_id: Unique file identifier
            user_id: User who owns this data
        
        Returns:
            Redacted dataframe
        """
        redacted_df = df.copy()
        redaction_count = 0
        
        for col in redacted_df.columns:
            if redacted_df[col].dtype == 'object':  # Text columns only
                redacted_df[col], col_count = self._redact_column(
                    redacted_df[col], 
                    file_id, 
                    col, 
                    user_id
                )
                redaction_count += col_count
        
        # Log redaction action
        self._log_action('REDACT_DATAFRAME', None, user_id, {
            'file_id': file_id,
            'rows': len(df),
            'columns': len(df.columns),
            'redactions': redaction_count
        })
        
        print(f"🔒 Redacted {redaction_count} PII values from {file_id}")
        return redacted_df
    
    def _redact_column(self, series: pd.Series, file_id: str, col_name: str, user_id: str):
        """Redact PII from a single column"""
        redaction_count = 0
        
        def redact_value(value):
            nonlocal redaction_count
            if pd.isna(value):
                return value
            
            result, count = self._redact_text(str(value), file_id, col_name, user_id)
            redaction_count += count
            return result
        
        redacted_series = series.apply(redact_value)
        return redacted_series, redaction_count
    
    def _redact_text(self, text: str, file_id: str, col_name: str, user_id: str):
        """
        Redact PII from single text value
        
        Returns:
            (redacted_text, redaction_count)
        """
        if not text or text == 'nan':
            return text, 0
        
        # Analyze for PII
        results = self.analyzer.analyze(
            text=text,
            language='en',
            entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "PERSON"]
        )
        
        if not results:
            return text, 0
        
        # Replace each PII entity
        redacted_text = text
        redaction_count = 0
        
        for result in sorted(results, key=lambda x: x.start, reverse=True):
            entity_type = result.entity_type
            original_value = text[result.start:result.end]
            
            # Create unique token
            token = self._create_token(original_value, entity_type, file_id, col_name)
            
            # Encrypt and store
            self._store_mapping(
                token=token,
                original_value=original_value,
                entity_type=entity_type,
                user_id=user_id,
                file_id=file_id,
                column_name=col_name
            )
            
            # Replace in text
            redacted_text = redacted_text[:result.start] + token + redacted_text[result.end:]
            redaction_count += 1
        
        return redacted_text, redaction_count
    
    def _create_token(self, value: str, entity_type: str, file_id: str, col_name: str) -> str:
        """Create unique token for PII value"""
        hash_input = f"{file_id}_{col_name}_{value}".encode()
        token_id = hashlib.sha256(hash_input).hexdigest()[:12]  # 12 chars for uniqueness
        return f"[{entity_type}_{token_id}]"
    
    def _store_mapping(self, token: str, original_value: str, entity_type: str, 
                      user_id: str, file_id: str, column_name: str):
        """Store encrypted PII mapping in database"""
        # Encrypt the original value
        encrypted_value = self.cipher.encrypt(original_value.encode())
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO pii_mappings 
            (token, encrypted_value, entity_type, user_id, file_id, column_name)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (token, encrypted_value, entity_type, user_id, file_id, column_name))
        
        conn.commit()
        conn.close()
    
    def restore(self, text: str, user_id: str, require_auth: bool = True) -> str:
        """
        Restore original PII values in text
        
        Args:
            text: Text with redacted tokens
            user_id: User requesting restoration
            require_auth: If True, only restore for authorized user
        
        Returns:
            Text with PII restored
        """
        if not text:
            return text
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all mappings for this user
        if require_auth:
            cursor.execute('''
                SELECT token, encrypted_value 
                FROM pii_mappings 
                WHERE user_id = ?
            ''', (user_id,))
        else:
            # Admin override - restore all
            cursor.execute('SELECT token, encrypted_value FROM pii_mappings')
        
        restored_text = text
        restore_count = 0
        
        for token, encrypted_value in cursor.fetchall():
            if token in text:
                try:
                    # Decrypt original value
                    original_value = self.cipher.decrypt(encrypted_value).decode()
                    
                    # Replace token with original
                    restored_text = restored_text.replace(token, original_value)
                    restore_count += 1
                    
                    # Update access stats
                    self._update_access_stats(token)
                    
                except Exception as e:
                    print(f"⚠️  Failed to decrypt token {token}: {e}")
                    continue
        
        conn.close()
        
        # Log restoration
        if restore_count > 0:
            self._log_action('RESTORE_PII', None, user_id, {
                'restored_count': restore_count
            })
            print(f"🔓 Restored {restore_count} PII values for user {user_id}")
        
        return restored_text
    
    def _update_access_stats(self, token: str):
        """Update access count and timestamp for a token"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE pii_mappings 
            SET accessed_count = accessed_count + 1,
                last_accessed = CURRENT_TIMESTAMP
            WHERE token = ?
        ''', (token,))
        
        conn.commit()
        conn.close()
    
    def _log_action(self, action: str, token: Optional[str], user_id: str, 
                   details: dict, ip_address: str = None):
        """Log action to audit trail"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_log (action, token, user_id, ip_address, details)
            VALUES (?, ?, ?, ?, ?)
        ''', (action, token, user_id, ip_address, json.dumps(details)))
        
        conn.commit()
        conn.close()
    
    def cleanup_old_mappings(self, days: int = 30, user_id: Optional[str] = None):
        """
        Delete PII mappings older than X days
        
        Args:
            days: Age threshold in days
            user_id: If provided, only cleanup for this user
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        if user_id:
            cursor.execute('''
                DELETE FROM pii_mappings 
                WHERE created_at < ? AND user_id = ?
            ''', (cutoff_date, user_id))
        else:
            cursor.execute('''
                DELETE FROM pii_mappings 
                WHERE created_at < ?
            ''', (cutoff_date,))
        
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        print(f"🗑️  Cleaned up {deleted_count} old PII mappings (>{days} days)")
        
        # Log cleanup
        self._log_action('CLEANUP', None, user_id or 'SYSTEM', {
            'days': days,
            'deleted_count': deleted_count
        })
        
        return deleted_count
    
    def get_user_mappings_count(self, user_id: str) -> int:
        """Get count of stored PII mappings for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM pii_mappings WHERE user_id = ?
        ''', (user_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def get_audit_log(self, user_id: Optional[str] = None, limit: int = 100):
        """Get audit log entries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute('''
                SELECT action, token, user_id, timestamp, details
                FROM audit_log
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (user_id, limit))
        else:
            cursor.execute('''
                SELECT action, token, user_id, timestamp, details
                FROM audit_log
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
        
        logs = cursor.fetchall()
        conn.close()
        
        return logs
    
    def export_user_data(self, user_id: str, output_file: str):
        """Export all PII mappings for a user (GDPR compliance)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT token, entity_type, file_id, column_name, created_at, accessed_count
            FROM pii_mappings
            WHERE user_id = ?
        ''', (user_id,))
        
        data = []
        for row in cursor.fetchall():
            data.append({
                'token': row[0],
                'entity_type': row[1],
                'file_id': row[2],
                'column_name': row[3],
                'created_at': row[4],
                'accessed_count': row[5]
            })
        
        conn.close()
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📦 Exported {len(data)} PII records for user {user_id} to {output_file}")
        
        return len(data)
    
    def delete_user_data(self, user_id: str):
        """Delete all PII mappings for a user (GDPR right to deletion)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM pii_mappings WHERE user_id = ?', (user_id,))
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        print(f"🗑️  Deleted all {deleted_count} PII records for user {user_id}")
        
        # Log deletion
        self._log_action('DELETE_USER_DATA', None, user_id, {
            'deleted_count': deleted_count
        })
        
        return deleted_count
    
    def get_statistics(self):
        """Get overall statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total mappings
        cursor.execute('SELECT COUNT(*) FROM pii_mappings')
        total_mappings = cursor.fetchone()[0]
        
        # Mappings by type
        cursor.execute('''
            SELECT entity_type, COUNT(*) 
            FROM pii_mappings 
            GROUP BY entity_type
        ''')
        by_type = dict(cursor.fetchall())
        
        # Total users
        cursor.execute('SELECT COUNT(DISTINCT user_id) FROM pii_mappings')
        total_users = cursor.fetchone()[0]
        
        # Total audit logs
        cursor.execute('SELECT COUNT(*) FROM audit_log')
        total_logs = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_mappings': total_mappings,
            'mappings_by_type': by_type,
            'total_users': total_users,
            'total_audit_logs': total_logs
        }


# Global instance
secure_pii_redactor = SecurePIIRedactor()