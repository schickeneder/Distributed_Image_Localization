pip freeze > requirements.txt
docker-compose up -d --build
./manage.py startapp taskapp
docker exec -it django /bin/sh
docker exec -it flask /bin/sh
docker-compose --env-file local_dev.env up -d --build 
docker-compose -f docker-compose-remote.yml --env-file remote_dev.env up -d --build 
# run tileserver in C:\Users\ps\Helium_Data\tileserver data where config.json, style.json and .mbtiles are
docker run -it -v "$(pwd):/data" -p 8080:8080 maptiler/tileserver-gl
