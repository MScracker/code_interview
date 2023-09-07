# -*- coding: utf-8 -*- 
# @since : 2020/7/29 14:53 
# @author : wongleon

# -*- coding:utf-8 -*-

# 类函数和静态函数
class People(object):
    # 类变量
    total = 0

    def __init__(self, name, age, country="US"):
        # 调用父类的初始化函数
        super(People, self).__init__()
        # 初始化当前类对象的一些属性
        self.name = name
        self.age = age
        self.country = country

    # 对象函数 只能由对象调用
    def eat(self):
        print('该吃饭了。。。')

    # 类函数  不用声明对象便可调用
    # 装饰器是以@开头，@结构的称之为语法糖,装饰器的作用主要是给一些现有的函数添加一些额外的功能
    @classmethod
    def work(cls, time, *args, **kwargs):
        # cls class 如果是类调用该函数，cls指的是这个类
        # 如果是对象调用该函数，cls指得就是这个对象的类型
        print(cls)

    @classmethod
    def sleep(cls):
        print('每一个类函数前必须添加装饰器 @classmethod')

    # 静态函数
    # @staticmethod  描述的函数称为静态函数，静态函数可以由类和对象调用，函数中没有隐形参数
    @staticmethod
    def run(time):
        print('跑步%s分钟....' % time)



class Tom(People):

    def __init__(self, name, age, gender, **kwargs):
        super().__init__(name, age, **kwargs)
        self.gender = gender



# # 对象函数只能由对象调用
# # 类函数由类调用、也可以用对象调用
# People.work(10)
# p1 = People('张三', 22)
# p1.work(10)

# # 调用静态函数
# # 类调用静态函数
# People.run(100)
# # 对象调用静态函数
# p1.run(50)

tom = Tom('tom', 30, 'F')
print()