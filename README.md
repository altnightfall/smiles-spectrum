# smiles-spectrum

**smiles-spectrum** — это инструмент для генерации масс-спектров на основе SMILES-строк молекул. 
Он предназначен для использования в химической аналитике, а также для образовательных целей.

#### Warning: на данный момент визуализируется приблизительный спект, который не может считаться точным рассчетом. 

## Установка

### Клонирование репозитория

```bash
git clone https://github.com/altnightfall/smiles-spectrum.git
cd smiles-spectrum
```
### Установка зависимостей
```bash
uv sync
```
## Использование
### Docker

```bash
make deploy
```

### Host

#### Backend
```bash
uv run uvicorn backend.app.main:app --reload --port 8000
```
#### Frontend
```bash
uv run streamlit run .\frontend\app.py
```
#### Тесты
```bash
uv run coverage run -m pytest
```