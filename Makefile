DC ?= docker-compose

.PHONY: all build up down deploy clean logs

all: build

build:
	$(DC) build --parallel

up:
	$(DC) up -d

down:
	$(DC) down

logs:
	$(DC) logs -f

deploy: build up
	@echo "✅ Сервисы развернуты (docker compose up -d)."

clean:
	$(DC) down --rmi all --volumes --remove-orphans
	@echo "🧹 Очистка завершена."
