def convert_percent(percent, screen, sensibility): 
        distance_from_50 = (percent - 50) / 50
        base_value = (percent / 100 * screen)
        k = sensibility
        converted_value = base_value * (1 + k * distance_from_50)

        #suavização (falta [smooth_value = 5] <- no param da função) 
            # filtered_values = np.convolve(converted_value, np.ones(smooth_value) / smooth_value, mode='valid')
            # final_values = np.clip(filtered_values, 0, screen)
            # return final_values
            
        if (0 >= converted_value):
             return 0
        elif (converted_value > screen): 
             return screen - 1
        else: 
            return int(converted_value)
