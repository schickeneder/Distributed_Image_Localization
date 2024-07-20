#!/bin/ash
# linux alpine runs ash rather than bash

echo "Apply database migrations"
python manage.py migrate

exec "$@"