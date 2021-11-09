# parkinson_flask_linebot
## How to use Docker on 201
0. execute ngrok and edit the URL in run.py and https://developers.line.biz/console/channel/xxxxx/messaging-api
```bash
./ngrok http 15003
```
1. in the directory of Dockerfile
```bash
docker build . -t parkinson_line  
docker run -itd --rm --name parkinson_linebot -v /home/jeremylai/docker_projects/parkinson_line/app:/workspace/parkinson -p 15003:15003 parkinson_line
```
2. go into the running container bash  
```bash
docker exec -it [container ID] bash
```
3. start the flask  
```bash
python3 bin/run.py
```
## the API will work on this URL
https://api.openaifab.com:15003/  
https://xxxxxx.ngrok.io/callback  
