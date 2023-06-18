FROM --platform=linux/amd64 ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y libvips-dev apache2 apache2-utils python3 python3-pip
RUN apt clean

COPY . /app
RUN rm -rf /var/www/html
RUN ln -sf /app/web /var/www/html
RUN chown www-data:www-data /var/www/html -R
RUN chmod 777 /var/www/html -R
RUN usermod -aG www-data root

EXPOSE 80 443
CMD ["apache2ctl", "-D", "FOREGROUND"]







