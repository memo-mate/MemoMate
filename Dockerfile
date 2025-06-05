FROM localhost:5009/easyfin/backend:fcd2f8c9

WORKDIR /app/

COPY . .

CMD ["fastapi", "run", "--workers", "4", "app/main.py"]