cp run/Dockerfile ./
cp -r ../tic-toc-toe.py ./app.py
docker build -t my-rl:latest .
NAME="AlexeyRL"
docker rm "${NAME}"
docker run -p 8888:8888 -e MPLBACKEND="module://itermplot" --name="${NAME}" -i -t my-rl
