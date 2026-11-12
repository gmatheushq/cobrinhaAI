import importlib.util


### Testador de bibliotecas

def is_installed(module_name):
    """Verifica se um módulo está instalado."""
    spec = importlib.util.find_spec(module_name)
    return spec is not None

# Exemplo de uso
biblioteca = "pygame"
if is_installed(biblioteca):
    print(f"{biblioteca} está instalado.")
else:
    print(f"{biblioteca} NÃO está instalado.")