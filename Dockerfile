FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

RUN apt update
RUN apt install -y cmake unzip libvips-dev apache2 apache2-utils python3 python3-pip ffmpeg libsm6 libxext6
RUN apt clean

COPY . /app
RUN rm -rf /var/www/html
RUN ln -sf /app/web /var/www/html
RUN chown www-data:www-data /var/www/html -R
RUN chmod 777 /var/www/html -R
RUN usermod -aG www-data root
RUN pip install -r /app/requirements.txt
RUN mkdir /app/web/data/
WORKDIR "/app"

EXPOSE 80 443
CMD ["apache2ctl", "-D", "FOREGROUND"]







