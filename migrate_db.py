#!/usr/bin/env python3
"""
Database migration script to add new columns for Ultravox built-in tools support
"""
import sqlite3
import os

DB_PATH = "outbound_agents_v2.db"

def migrate():
    print("🔄 Starting database migration...")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(tools)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Add is_builtin column to tools table if it doesn't exist
        if 'is_builtin' not in columns:
            print("  Adding is_builtin column to tools table...")
            cursor.execute("ALTER TABLE tools ADD COLUMN is_builtin BOOLEAN DEFAULT 0")
            print("  ✅ Added is_builtin column")
        else:
            print("  ⏭️  is_builtin column already exists")
        
        # Add parameter_overrides column to tools table if it doesn't exist
        if 'parameter_overrides' not in columns:
            print("  Adding parameter_overrides column to tools table...")
            cursor.execute("ALTER TABLE tools ADD COLUMN parameter_overrides TEXT")
            print("  ✅ Added parameter_overrides column")
        else:
            print("  ⏭️  parameter_overrides column already exists")
        
        # Check agents table
        cursor.execute("PRAGMA table_info(agents)")
        agent_columns = [col[1] for col in cursor.fetchall()]
        
        # Add transfer_number column to agents table if it doesn't exist
        if 'transfer_number' not in agent_columns:
            print("  Adding transfer_number column to agents table...")
            cursor.execute("ALTER TABLE agents ADD COLUMN transfer_number TEXT")
            print("  ✅ Added transfer_number column")
        else:
            print("  ⏭️  transfer_number column already exists")
        
        conn.commit()
        print("✅ Migration completed successfully!")
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    if not os.path.exists(DB_PATH):
        print(f"❌ Database file {DB_PATH} not found!")
        exit(1)
    
    migrate()
