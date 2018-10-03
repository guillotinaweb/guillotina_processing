All this exercice is counting on having a k8s cluster on the current context.
You can use docker for desktop k8s context.

Run It
------


docker-compose -f local/docker-compose.yaml up

open browser at http://localhost:8888

Jupyter default password: ```grocks```


Clean up
--------

docker-compose -f local/docker-compose.yaml stop
docker-compose -f local/docker-compose.yaml rm -v