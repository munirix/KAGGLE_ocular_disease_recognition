"""
Pacote `src` do TCC:
"Aplicacao de Inteligencia Artificial para diagnostico de defeitos refrativos
atraves de imagens oftalmologicas".

Pipeline em 4 etapas sobre o dataset ODIR-5K:
  Etapa 1 - pre-processamento de imagens de resolucao variada (preprocessing.py)
  Etapa 2 - classificacao Normal x Miopia             (stage2_classification.py)
  Etapa 3 - severidade leve/alta/magna                (stage3_severity.py)
  Etapa 4 - regressao do grau (dioptrias)             (stage4_regression.py)

Ver README/analise para o aviso metodologico sobre os alvos sinteticos das
Etapas 3 e 4.
"""
__version__ = "1.0.0"