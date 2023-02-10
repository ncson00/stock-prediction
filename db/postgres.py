import psycopg2
import psycopg2.extras as extras

PG_USER = "postgres"
PG_PASSWORD = "1"
PG_DATABASE = "stock"
PG_HOST = "172.18.0.3"
PG_PORT = 5432
PG_DRIVER = 'org.postgresql.Driver'


def connect():
    try:
        conn = psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD,
        )
        print("Connect Postgres Successfully!!")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        return

    return conn


def write_postgres(conn, df, table_name):
    """
    Using psycopg2.extras.execute_values() to insert the dataframe
    """
    # Create a list of tupples from the dataframe values
    tuples = [tuple(x) for x in df.to_numpy()]
    # Comma-separated dataframe columns
    cols = ','.join(list(df.columns))
    # SQL quert to execute
    query = "INSERT INTO %s(%s) VALUES %%s" % (table_name, cols)
    cursor = conn.cursor()
    try:
        extras.execute_values(cursor, query, tuples)
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        cursor.close()
        return 1
    print("execute_values() done")
    cursor.close()