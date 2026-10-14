# Diretrizes e Workflow do Auditor de Código Python (`python-auditor`)
**L.O.R.E.N.A.-DESKTOP — Governança Contínua de Qualidade, Clean Code e Garbage Cleanup**

---

## 1. Missão do Auditor

O **Auditor de Código Python (`python-auditor`)** atua como portão de segurança e governança contínua (Safety Gate) entre a implementação de qualquer feature/correção e a entrega final. Seu objetivo é garantir que o código adicionado seja estritamente limpo, aderente ao DDD, livre de sobre-engenharia e que nenhum resíduo de compilação ou lixo de build permaneça no repositório.

---

## 2. Checklist Obrigatório de Auditoria (Portão 6)

### A. Clean Code e SOLID
- **Nomes Revelam Intenção:** Identificadores devem responder *o que fazem e por que existem*. Proibido o uso de `data`, `data2`, `temp`, `val`, `res`, `obj` desqualificados.
- **Funções Concisas ($\le 25$ linhas):** Cada função deve operar em um único nível de abstração (SRP) e realizar apenas uma tarefa.
- **Limite de Parâmetros ($\le 4$ argumentos):** Se uma função requer 5+ argumentos, eles devem ser agrupados em um objeto de configuração ou dataclass tipada.
- **Zero Exceções Silenciosas:** NUNCA permita blocos `except: pass` ou `except Exception: pass`. Toda exceção deve ser tipada, tratada ou registrada com log apropriado.
- **Tipagem Estrita (Python 3.12+):** Todas as assinaturas de funções e métodos públicos devem conter *type hints* completos.

### B. Hot Loop & Tempo Real (30 FPS)
- **Zero Alocações no Loop Crítico:** Em rotinas que executam a cada frame do `TrackingPipeline` ($\approx 33\text{ ms}$), é proibido instanciar arrays NumPy ou listas dinâmicas. Reutilize buffers pré-alocados (`out=self._buf`).
- **Isolamento de Threads:** Nenhuma chamada bloqueante (I/O, OpenCV, `time.sleep`) é permitida na thread principal do Tkinter.

### C. Testabilidade (Pytest Standard)
- A suíte de testes deve rodar com 100% de sucesso (`.venv\Scripts\python -m pytest -q`).
- Proibido o uso de `time.sleep()` para sincronização de threads em testes (use `threading.Event`).
- Proibido qualquer bypass de código em produção (`if "pytest" in sys.modules`).
- Tolerâncias numéricas contínuas devem utilizar `pytest.approx` ou `np.allclose`.

---

## 3. Protocolo de Cleanup de Garbage Code e Resíduos de Build

O Auditor deve rodar ou verificar a execução desta rotina de limpeza antes de aprovar qualquer entrega:

```powershell
# 1. Purgar artefatos de build, distribuição e empacotamento
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue build, dist, *.egg-info, *.spec

# 2. Purgar caches de compilação Python (*.pyc e __pycache__)
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue

# 3. Purgar caches de cobertura e testes transitórios
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .pytest_cache, htmlcov, .coverage

# 4. Verificar arquivos de rascunho em scratch/
# Scripts que já viraram código de produção em lorena/ ou testes em tests/ devem ser revisados ou limpos.
```

---

## 4. Estrutura do Parecer de Auditoria

O parecer emitido pelo `python-auditor` deve conter:
1. **Status do Portão:** `[APROVADO]` ou `[BLOQUEADO]`;
2. **Resumo do Cleanup:** confirmação de limpeza de `build/`, `dist/`, `.spec` e caches;
3. **Tabela de Apontamentos:** categorizados em `[BLOCKER]`, `[WARNING]` e `[CLEANUP]`;
4. **Evidência Pytest:** contagem exata de testes aprovados sem falhas e sem skips.
