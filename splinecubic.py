


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import CubicSpline

# # Coordenadas virtuais normalizadas
# virtual_points = {
#     "0": [0.41380630651890726, 0.6416122004357298],
#     "1": [0.4263486312399356, 0.7695436507936507],
#     "2": [0.47708897195752603, 0.7860269360269361],
#     "3": [0.6063983965464076, 0.5394134846921225],
#     "4": [0.6283012050819636, 0.7160027472527473],
#     "5": [0.6186700896003221, 0.7767676767676769],
#     "6": [0.7949429540548874, 0.6559454191033139],
#     "7": [0.7862010484119644, 0.7217728758169936],
#     "8": [0.7641783281318165, 0.862037037037037]
# }

# # Coordenadas reais dos pontos
# real_points = {
#     "0": [0, 0],
#     "1": [0, 540],
#     "2": [0, 1080],
#     "3": [960, 0],
#     "4": [960, 540],
#     "5": [960, 1080],
#     "6": [1920, 0],
#     "7": [1920, 540],
#     "8": [1920, 1080]
# }

# # Usar coordenadas reais diretamente
# converted_x = [real_points[str(i)][0] for i in range(len(real_points))]
# converted_y = [real_points[str(i)][1] for i in range(len(real_points))]

# # Interpolação cúbica
# cs_x = CubicSpline(range(len(converted_x)), converted_x)
# cs_y = CubicSpline(range(len(converted_y)), converted_y)

# # Gerar valores interpolados
# n_points = 900
# interp_indices = np.linspace(0, len(converted_x) - 1, n_points)
# interp_x = cs_x(interp_indices)
# interp_y = cs_y(interp_indices)

# # Plotar os resultados
# plt.figure(figsize=(10, 6))
# plt.plot(interp_x, interp_y, label="Spline Cúbica", color="blue")
# plt.scatter(converted_x, converted_y, color="red", label="Pontos Reais")

# # Ajustar a visualização para a resolução da tela
# plt.xlim(0, 1920)
# plt.ylim(1080, 0)
# plt.xlabel("x (px)")
# plt.ylabel("y (px)")
# plt.title("Interpolação com Spline Cúbica")
# plt.legend()
# plt.grid()
# plt.show()
