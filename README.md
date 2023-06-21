# Requirements

1. Docker or Docker Desktop
2. 32Gb+ RAM
3. GPU (optional)

# Data Preparation

Suppose the data was stored in `/Users/username/data`. Please organize the directory as follows 
(Currently we only support the differential analysis for two groups of images):
```angular2html
Responders/
    case1.tif
    case2.tif
    ...
NonResponders/
    case1.tif
    case2.tif
    ...
```
Please make sure the filename does not contain special characters (e.g., `,`, `&`, etc.).

# Data Processing

When the data is ready, run the following command step by step:

```angular2html
docker pull 35.222.196.170:5000/debug:3.0

docker run -d -it --name test --memory="32g" --memory-swap="64g" -v "/Users/username/data:/appdata" -p 8080:80 35.222.196.170:5000/debug:3.0

docker exec -d -it test bash /app/run.sh /appdata/inputs myproject

```

# Data Visualization

After data processing completed, open http://localhost:8080 in browser to check the results.