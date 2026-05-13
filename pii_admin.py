"""
PII Admin Tool - Manage PII redaction system
"""

import argparse
from pii_redactor import secure_pii_redactor


def main():
    parser = argparse.ArgumentParser(description='PII Redaction Admin Tool')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Stats command
    subparsers.add_parser('stats', help='Show statistics')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Cleanup old mappings')
    cleanup_parser.add_argument('--days', type=int, default=30, help='Delete mappings older than N days')
    cleanup_parser.add_argument('--user', help='User ID (optional)')
    
    # Audit log command
    audit_parser = subparsers.add_parser('audit', help='Show audit log')
    audit_parser.add_argument('--user', help='User ID (optional)')
    audit_parser.add_argument('--limit', type=int, default=50, help='Number of entries')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export user data')
    export_parser.add_argument('--user', required=True, help='User ID')
    export_parser.add_argument('--output', required=True, help='Output file')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete user data')
    delete_parser.add_argument('--user', required=True, help='User ID')
    delete_parser.add_argument('--confirm', action='store_true', help='Confirm deletion')
    
    args = parser.parse_args()
    
    if args.command == 'stats':
        stats = secure_pii_redactor.get_statistics()
        print("\n📊 PII Redaction Statistics:")
        print(f"  Total mappings: {stats['total_mappings']}")
        print(f"  Total users: {stats['total_users']}")
        print(f"  Total audit logs: {stats['total_audit_logs']}")
        print(f"\n  Mappings by type:")
        for entity_type, count in stats['mappings_by_type'].items():
            print(f"    {entity_type}: {count}")
    
    elif args.command == 'cleanup':
        count = secure_pii_redactor.cleanup_old_mappings(args.days, args.user)
        print(f"✅ Cleaned up {count} old mappings")
    
    elif args.command == 'audit':
        logs = secure_pii_redactor.get_audit_log(args.user, args.limit)
        print(f"\n📋 Audit Log (last {args.limit} entries):")
        for action, token, user_id, timestamp, details in logs:
            print(f"  [{timestamp}] {action} by {user_id}: {details}")
    
    elif args.command == 'export':
        count = secure_pii_redactor.export_user_data(args.user, args.output)
        print(f"✅ Exported {count} records to {args.output}")
    
    elif args.command == 'delete':
        if not args.confirm:
            print("⚠️  Add --confirm to actually delete data")
            return
        
        count = secure_pii_redactor.delete_user_data(args.user)
        print(f"✅ Deleted {count} records for user {args.user}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()