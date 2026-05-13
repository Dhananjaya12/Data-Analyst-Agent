"""
CSV Registry - Manages multiple CSV files with rich metadata
"""

import pandas as pd
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class CSVFileInfo:
    """Metadata about a single CSV file"""
    file_id: str
    file_path: str
    file_name: str
    description: str
    df: pd.DataFrame = None  # Note: it's 'df' not 'dataframe'
    
    # Auto-generated metadata
    columns: List[str] = field(default_factory=list)
    column_types: Dict[str, str] = field(default_factory=dict)
    row_count: int = 0
    sample_rows: list = field(default_factory=list)
    column_stats: Dict[str, dict] = field(default_factory=dict)
    
    def build_context(self) -> str:
        """Build rich context for LLM / embedding"""
        col_details = []
        for col in self.columns:
            dtype = self.column_types.get(col, "unknown")
            stats = self.column_stats.get(col, {})
            
            if "range" in stats:
                detail = f"  - {col} ({dtype}): range {stats['min']} to {stats['max']}, avg {stats.get('mean', 'N/A')}"
            elif "values" in stats:
                detail = f"  - {col} ({dtype}): values {stats['values']}"
            else:
                detail = f"  - {col} ({dtype})"
            col_details.append(detail)
        
        return f"""FILE: {self.file_name} (id: {self.file_id})
DESCRIPTION: {self.description}
ROWS: {self.row_count}
COLUMNS:
{chr(10).join(col_details)}
SAMPLE DATA (first 2 rows):
{self.sample_rows[:2]}
"""


class CSVRegistry:
    """Manages multiple CSV files in a session"""
    
    def __init__(self):
        self.files: Dict[str, CSVFileInfo] = {}
    
    def register(self, file_path: str, description: str = "", file_id: str = None, user_id: str = "default_user") -> str:
        """
        Register a CSV file with PII redaction
        
        Args:
            file_path: Path to CSV file
            description: Optional description
            file_id: Optional custom ID
            user_id: User ID for PII isolation
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        df = pd.read_csv(file_path)
        file_name = os.path.basename(file_path)
        file_id = file_id or os.path.splitext(file_name)[0]
        
        # PII Redaction
        try:
            from pii_redactor import secure_pii_redactor
            print(f"🔒 Scanning {file_id} for PII...")
            df = secure_pii_redactor.redact_dataframe(df, file_id, user_id)
            print(f"✅ PII redaction complete")
        except ImportError:
            print(f"⚠️ PII redactor not available, skipping...")
        except Exception as e:
            print(f"⚠️ PII redaction failed: {e}")
        
        # Compute stats
        column_stats = self._compute_stats(df)
        sample_rows = df.head(2).to_dict('records')
        
        # Store file info - FIXED: use 'df' not 'dataframe'
        self.files[file_id] = CSVFileInfo(
            file_id=file_id,
            file_path=file_path,
            file_name=file_name,  # Added missing field
            description=description or f"Dataset: {file_id}",
            df=df,  # FIXED: was 'dataframe', should be 'df'
            columns=list(df.columns),
            row_count=len(df),
            column_types={col: str(dtype) for col, dtype in df.dtypes.items()},
            sample_rows=sample_rows,  # Added
            column_stats=column_stats  # Added
        )
        
        print(f"✅ Registered: {file_id} ({len(df)} rows, {len(df.columns)} cols)")
        return file_id
    
    def _compute_stats(self, df: pd.DataFrame) -> Dict[str, dict]:
        """Compute column statistics"""
        stats = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                stats[col] = {
                    "range": True,
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "mean": round(float(df[col].mean()), 2)
                }
            else:
                unique = df[col].unique()
                if len(unique) <= 10:
                    stats[col] = {"values": list(unique[:10])}  # Limit to 10
                else:
                    stats[col] = {"unique_count": len(unique), "examples": list(unique[:3])}
        return stats
    
    def get(self, file_id: str) -> Optional[CSVFileInfo]:
        return self.files.get(file_id)
    
    def list_files(self) -> List[CSVFileInfo]:
        return list(self.files.values())
    
    def get_all_contexts(self) -> str:
        """All file descriptions combined - for router agent"""
        return "\n\n".join(info.build_context() for info in self.files.values())
    
    def remove(self, file_id: str) -> bool:
        """Remove a file from the registry AND delete the backing CSV on disk."""
        if file_id not in self.files:
            return False
        info = self.files[file_id]
        try:
            if info.file_path and os.path.exists(info.file_path):
                os.remove(info.file_path)
                print(f"🗑️  Deleted file: {info.file_path}")
        except Exception as e:
            print(f"⚠️  Could not delete {info.file_path}: {e}")
        del self.files[file_id]
        print(f"🗑️  Unregistered: {file_id}")
        return True