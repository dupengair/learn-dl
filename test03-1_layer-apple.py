from common import MulLayer, AddLayer

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
print("forward") 
apple_price = mul_apple_layer.forward(apple, apple_num) #(1)
orange_price = mul_orange_layer.forward(orange, orange_num) #(2)
print(apple_price, orange_price) # 220
all_price = add_apple_orange_layer.forward(apple_price, orange_price) #(3)
price = mul_tax_layer.forward(all_price, tax) #(4)
print(all_price, price) # 220


# backward
print("backward") 
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice) #(4)
print(dall_price, dtax) # 220
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price) #(3)
print(dapple_price, dorange_price) # 220
dorange, dorange_num = mul_orange_layer.backward(dorange_price) #(2)
print(dorange, dorange_num) # 220
dapple, dapple_num = mul_apple_layer.backward(dapple_price) #(1)
print(dapple, dapple_num) # 220

print(price) # 715
print(dapple_num, dapple, dorange, dorange_num, dtax) # 110 2.2 3.3 165 650

