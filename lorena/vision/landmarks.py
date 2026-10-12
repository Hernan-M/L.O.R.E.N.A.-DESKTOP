"""Índices de landmarks do MediaPipe Face Mesh (478 pontos, com íris).

Os índices são ANATÔMICOS (olho esquerdo = olho esquerdo do usuário) quando a
detecção roda no frame não espelhado.
"""

# Cantos ordenados esquerda→direita da IMAGEM (frame não espelhado).
RIGHT_EYE_CORNERS = (33, 133)    # olho direito do usuário: externo, interno
LEFT_EYE_CORNERS = (362, 263)    # olho esquerdo do usuário: interno, externo

# 6 pontos para o EAR: p1 (canto), p2, p3 (pálpebra sup.), p4 (canto), p5, p6 (pálpebra inf.)
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR = (362, 385, 387, 263, 373, 380)

# Centros de íris (a atribuição a cada olho é verificada por proximidade).
IRIS_CENTERS = (468, 473)
IRIS_RING_A = (469, 470, 471, 472)
IRIS_RING_B = (474, 475, 476, 477)

NUM_LANDMARKS_WITH_IRIS = 478
