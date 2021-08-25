import sqlite3


def create_database(db_file):
    """Create a connection to a SQLite database."""
    connection = None
    try:
        connection = sqlite3.connect(db_file)
        print(f'Connected to {db_file} (sqlite3 version = {sqlite3.version})')
    except sqlite3.Error as err:
        print(err)

    if connection:
        connection.close()
        return True
    return False
