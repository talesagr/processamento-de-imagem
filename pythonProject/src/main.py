def check_input(color):
    if 0 >= color >= 255:
        print("ERRO digite um numero de 0 a 255")
        exit()


def check_h_hsv(r, g, b):
    h = 0
    max = find_bigger(r, g, b)
    min = find_smaller(r, g, b)
    if max == min:
        h = 0
    elif max == r and g >= b:
        h = (60 * ((g - b) / (max - min))) + 0
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
    max = find_bigger(r, g, b)
    min = find_smaller(r, g, b)
    if max == 0:
        return 0
    return (max - min) / max


def normalize_rgb(r, g, b):
    normalized_r = r / (r + g + b)
    normalized_g = g / (r + g + b)
    normalized_b = b / (r + g + b)

    print(f"Cores normalizadas em RGB: ", "R: " "{:.2f}".format(normalized_r), "G: " "{:.2f}".format(normalized_g),
          "B: " "{:.2f}".format(normalized_b))


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
# Conversao RGB para HSV
h_hsv = check_h_hsv(r, g, b)
s_hsv = check_s_hsv(r, g, b)
v_hsv = find_bigger(r, g, b) / 255


print(f"HSV: ", "H: " "{:.2f}".format(h_hsv), "S: ""{:.2f}".format(s_hsv), "V: ", v_hsv)


def convert_hsv_to_rgb(h_hsv, s_hsv, v_hsv):
    c = v_hsv * s_hsv
    x = c * (1 - abs((h_hsv / 60) % 2 - 1))
    m = v_hsv - c
    if 0 <= h_hsv < 60:
        r_prime=c
        g_prime=x
        b_prime=0
    elif 60 <= h_hsv < 120:
        r_prime=x
        g_prime=c
        b_prime=0
    elif 120 <= h_hsv < 180:
        r_prime=0
        g_prime=c
        b_prime=x
    elif 180 <= h_hsv < 240:
        r_prime= 0
        g_prime=x
        b_prime=c
    elif 240 <= h_hsv < 300:
        r_prime=x
        g_prime=0
        b_prime=c
    elif 300 <= h_hsv < 360:
        r_prime=c
        g_prime=0
        b_prime=x
    else:
        r_prime=0
        g_prime=0
        b_prime=0

    r = (r_prime+m)*255
    g = (g_prime+m)*255
    b = (b_prime+m)*255

    print(f"R: {int(r)} G: {int(g)} B: {int(b)}")


convert_hsv_to_rgb(h_hsv, s_hsv, v_hsv)
# 255, 128, 100


def convert_rgb_to_cmyk(r, g, b):
    r_prime = r/255
    g_prime = g/255
    b_prime = b/255
    k = 1-find_bigger(r_prime,g_prime,b_prime)
    c = (1-r_prime-k)/(1-k)
    m = (1-g_prime-k)/(1-k)
    y = (1-b_prime-k)/(1-k)
    m_str = "{:.2f}".format(m)
    y_str = "{:.2f}".format(y)
    k_str = "{:.2f}".format(k)
    c_str = "{:.2f}".format(c)
    print(f"C: {c_str}  M: {m_str}  Y: {y_str}  K: {k_str}")
    return c, m, y, k


c,m,y,k = convert_rgb_to_cmyk(r,g,b)


def convert_cmyk_to_rgb(c, m, y, k):
    r = 255 * ((1-c) * (1-k))
    g = 255 * ((1-m) * (1-k))
    b = 255 * ((1-y) * (1-k))

    print(f"R: {int(r)} G: {int(g)} B: {int(b)}")


convert_cmyk_to_rgb(c,m,y,k)


def convert_rgb_to_grey_scale(r, g, b):
    grey = (r+g+b) / 3
    print(f"GREYSCALE: {grey}")


convert_rgb_to_grey_scale(r,g,b)
