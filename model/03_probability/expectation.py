def mean(x_list):
    if not x_list:
        raise ValueError("输入列表不能为空")
    return sum(x_list) / len(x_list)


def main():
    try:
        user_input = input("请输入若干实数，用空格分隔: ")
        x_list = [float(x) for x in user_input.split()]

        if not x_list:
            print("错误：未输入任何数字")
            return

        result = mean(x_list)
        print(f"平均值 E(x) = {result}")

    except ValueError as e:
        print(f"输入错误：请确保输入的都是实数。{e}")


if __name__ == "__main__":
    main()