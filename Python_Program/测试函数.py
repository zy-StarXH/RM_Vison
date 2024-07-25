def Position(box): # 在盒子中遍历坐标，将每个坐标排序
    left_up_pos = box[0]
    right_up_pos = []
    right_down_pos = []
    left_down_pos = []
    for x, y in range(box):
        if x <= left_up_pos[0] and y <= left_up_pos[1]:
            left_up_pos = [x, y]
        if x > left_up_pos[0] and y <= left_up_pos[1]:
            right_up_pos = [x, y]
        if x <= left_up_pos[0] and y > left_up_pos[1]:
            left_down_pos = [x, y]
        if x > left_up_pos[0] and y > left_up_pos[1]:
            right_down_pos = [x, y]
    return left_up_pos, right_up_pos, right_down_pos, left_down_pos

def main():
    box = []
    for i in range(4):
        a1 = eval(input('Enter x and y coordinates for position {} (e.g., 1, 2): '.format(i+1)))
        if len(a1) != 2 or not all(isinstance(coord, (int, float)) for coord in a1):
            print("Invalid input. Please enter two numbers for each position.")
            continue
        box.append(a1)
    corners = Position(box)
    print(corners)


main()