def check_input(color):
    print(color)
    if 0 >= color >= 255:
        print("ERRO digite um numero de 0 a 255")
        exit()

def check_h_hsv(r,g,b):
    h = 0
    max = find_bigger(r,g,b)
    min = find_smaller(r,g,b)
    if max == r and g >= b:
        h = (60 * ((g - b) / (max - min)) )+ 0
    elif max == r and g < b:
        h = (60 * ((g - b) / (max - min))) + 360
    elif max == g:
        h = (60 * ((b - r) / (max - min))) + 120
    elif max == b:
        h = (60 * ((r - g) / (max - min))) + 240
    return h


def find_bigger(r, g, b):
    if r > g:
        if r > b:
            return r
        else:
            return b
    else:
        if g > b:
            return g
        else:
            return b

def find_smaller(r, g, b):
    if r < g:
        if r < b:
            return r
        else:
            return b
    else:
        if g < b:
            return g
        else:
            return b


def check_s_hsv(r, g, b):
    max = find_bigger(r,g,b)
    min = find_smaller(r, g, b)
    return (max - min) / max

def normalize_rgb(r, g, b):
    normalized_r = r / (r + g + b)
    normalized_g = g / (r + g + b)
    normalized_b = b / (r + g + b)

    print(f"Cores normalizadas em RGB: ", "{:.2f}".format(normalized_r), "{:.2f}".format(normalized_g), "{:.2f}".format(normalized_b))

normalized_r = 0
normalized_g = 0
normalized_b = 0

print("*** PRIMEIRO TDE ***")
r = int(input("Digite o numero de R -> "))
check_input(r)
g = int(input("Digite o numero de G -> "))
check_input(g)
b = int(input("Digite o numero de B -> "))
check_input(b)

normalize_rgb(r, g, b)
h_hsv = check_h_hsv(r,g,b)
s_hsv = check_s_hsv(r,g,b)
v_hsv = find_bigger(r,g,b)

print(f"HSV: ", "{:.2f}".format(h_hsv), "{:.2f}".format(s_hsv), v_hsv)


def convert_hsv_to_rgb(h_hsv, s_hsv, v_hsv):
    c = v_hsv * s_hsv
    h = h_hsv % 360
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v_hsv - c



convert_hsv_to_rgb(h_hsv,s_hsv,v_hsv)
# 255, 128, 100