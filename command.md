pip freeze > requirements.txt
docker-compose up -d --build
./manage.py startapp taskapp
docker exec -it django /bin/sh