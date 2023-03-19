ROBOT_NUM = 4
WAITING_NUM = 8
ROBOT_MAX_SPEED = 6
INIT_SPEED = [0, 0, 0, 0]
INTERACTION_RADIUS = 0.4
ROBOT_RADIUS = 0.45
LOWEST_PRIORITY = 0.01
ALL_VALUE = 250000
NEEDED_BIN = [0b00000000, 0b00000000, 0b00000000, 0b00000110, 0b00001010, 0b00001100,
              0b01110000, 0b10000000, 0b11111110]
NEEDED_DEC = [[], [], [], [1, 2], [1, 3], [2, 3], [4, 5, 6], [7], list(range(1, 8))]
WORK_CYCLE = [50, 50, 50, 500, 500, 500, 1000, 1, 1]
DIRECT_NEXT = [[4, 5, 9], [4, 6, 9], [5, 6, 9], [7, 9], [7, 9], [7, 9], [8, 9], list(range(1, 7)), list(range(1, 7))]
PRICE = [[3000, 6000], [4400, 7600], [5800, 9200], [15400, 22500],
         [17200, 25000], [19200, 27500], [76000, 105000]]  # price for each product
