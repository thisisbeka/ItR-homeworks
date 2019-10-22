import numpy as np

arr = np.array([[9.99997459e-01, -1.59518945e-03, -1.59265292e-03, -1.38698743e+03],
                [-1.59518744e-03, -9.99998728e-01, 2.53452662e-06, 2.20844702e+00],
                [-1.59265494e-03, 6.05975094e-09, -9.99998732e-01, -1.76265013e+02],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


def rot2eul(r):
    if (r[0, 2] == 1) | (r[0, 2] == -1):
        c = 0
        dlt = np.arctan2(r[0, 1], r[1, 2])

        if r[0, 2] == -1:
            b = np.pi / 2
            a = c + dlt
        else:
            b = -np.pi / 2
            a = -c + dlt
    else:
        b = -np.arcsin(r[0, 2])
        print(b)
        a = np.arctan2(r[1, 2] / np.cos(b), r[2, 2] / np.cos(b))
        c = np.arctan2(r[0, 1] / np.cos(b), r[0, 0] / np.cos(b))
    return [a, b, c]


print(rot2eul(arr))
