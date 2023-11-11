up:
	docker compose up --build

down:
	docker compose down -v

logs:
	docker compose logs -f --tail=200

dash:
	open http://localhost:8501

spark-ui:
	open http://localhost:8080
