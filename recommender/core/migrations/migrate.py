import os

from yoyo import read_migrations
from yoyo import get_backend

from recommender.config.base import DB_NAME, DB_USER, DB_PASSWORD


def apply():
    backend = get_backend(f'postgres://{DB_USER}:{DB_PASSWORD}@db:5432/{DB_NAME}')
    migrations = read_migrations(os.path.dirname(os.path.abspath(__file__)))
    with backend.lock():
        backend.apply_migrations(backend.to_apply(migrations))


def rollback():
    backend = get_backend(f'postgres://{DB_USER}:{DB_PASSWORD}@db:5432/{DB_NAME}')
    migrations = read_migrations(os.path.dirname(os.path.abspath(__file__)))
    with backend.lock():
        backend.rollback_migrations(backend.to_rollback(migrations))
