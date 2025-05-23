import mysql.connector
from flask import g


def connect_db():
    return mysql.connector.connect(
        host="localhost", user="root", password="", database="startex_feedback"
    )


def get_db():
    if "db" not in g:
        g.db = connect_db()
    return g.db


def close_db(e=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()
