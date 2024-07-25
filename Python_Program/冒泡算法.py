def BubbleSort_X(arr):
    for i in range(1, len(arr)):
        for j in range(0, len(arr)-i):
            if arr[j][0] > arr[j+1][0]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def BubbleSort_Y(arr):
    for i in range(1, len(arr)):
        for j in range(0, len(arr)-i):
            if arr[j][1] > arr[j+1][1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr


def main():
    left = (3, 2)
    print(left[0])
    box = [(3, 1), (4, 5), (1, 4), (1, 1)]
    box = BubbleSort_X(box)
    box = BubbleSort_Y(box)
    print(box)

main()
