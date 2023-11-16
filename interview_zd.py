def max_area(height):
    left = 0
    right = len(height) - 1
    max_water = 0

    while left < right:
        # 计算当前两个指针指向的高度构成的容器的水量
        current_height = min(height[left], height[right])
        current_width = right - left
        current_water = current_height * current_width

        # 更新最大水量
        max_water = max(max_water, current_water)

        # 移动指针
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1

    return max_water


# 示例输入
height = [2, 2]

# 计算最大容纳水量
result = max_area(height)

print(result)
