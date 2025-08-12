#!/usr/bin/env python3

from database import db_manager
from sqlalchemy import text

print('Testing database connection...')
print('Engine:', db_manager.get_engine())
print('SessionLocal:', db_manager.SessionLocal)

if db_manager.get_engine() is None:
    print('ERROR: Engine is None!')
else:
    print('Engine is available')

if db_manager.SessionLocal is None:
    print('ERROR: SessionLocal is None!')
else:
    print('SessionLocal is available')

try:
    with db_manager.get_session() as session:
        if session is None:
            print('ERROR: Session is None!')
        else:
            result = session.execute(text('SELECT 1 as test'))
            print('Test query result:', result.fetchone())
            print('SUCCESS: Database connection working!')
except Exception as e:
    print(f'ERROR: Database connection failed: {e}')