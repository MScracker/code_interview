def binary_search_left_insert(nums: [int], target: int) -> int:
    left = 0
    right = len(nums) - 1

    while left < right:
        mid = (right + left) // 2
        if target > nums[mid]:
            left = mid + 1 #寻找区间[mid+1, right]
        elif target == nums[mid]:
            right = mid #查找区间[left, mid], 右边界包含目标值
        elif target < nums[mid]:
            right = mid - 1 #查找区间[left, mid - 1]

    if nums[left] == target:
        return left #最左边界的目标值
    return -1


def binary_search_right_insert(nums: [int], target: int) -> int:
    left = 0
    right = len(nums) - 1

    while left < right:
        mid = (left + right + 1) // 2
        if target > nums[mid]:
            left = mid + 1 #查找区间[mid+1, right]
        elif target == nums[mid]:
            left = mid #查找区间[mid, right], 左边界包含目标值
        elif target < nums[mid]:
            right = mid - 1 #查找区间[left, mid - 1], 包含目标值
    if nums[right] == target:
        return right #最右边界的目标值
    return -1


if __name__ == "__main__":
    idx = binary_search_right_insert([5,7,7,8,8,10], 8)
    print(idx)
