# name = 'Marchel'
# words = 'Hello guys'
# print(f'{name} : "{words}"')


# phi = 22/7
# print(f'{phi: .50f}')

# # List 
# my_list = []
# print(type(my_list))

# my_list.append([1,3, 5,4,7])
# my_list.append([3, 7,1])
# my_list.append(1)
# my_list.append(14)
# my_list.append(12)
# my_list.append([1,3,4,7])

# print(my_list)

# print(5*"=")

# for obj in my_list:
#     if isinstance(obj, type(my_list)):
#         for item in obj:
#             print({item})
#     else:
#         print(obj)

# # TUpe 
# my_tuple = ('a', 'b', 'c')
# q, w, e = my_tuple
# print(my_tuple)
# print(q,w,e)
# print(my_tuple[0])
# # my_tuple[0] = 't' # Cant be assigned 


# # Set 
# my_set = {1, 1, 1, 2, 2, 3, 3, 3, 3}
# print(my_set)

# # Dictionary 
# my_dict = {
#     'key1': 'Hello',
#     'key2': 'world',
#     'key3': 10
# }

# print(my_dict['key2'])
# print(my_dict.items())

# import cv2
# import numpy as np

# def show_result(winname = None, image = None): 
#     cv2.imshow(winname, image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# image = cv2.imread('./lena.jpg')

# show_result("Lena Image", image)

# # Starndarnya gambar itu RGB, namun ketika dibaca cv2, dia akan jadi BGR urutan channelnyaa
# image_b = image.copy()
# image_g = image.copy()
# image_r = image.copy()

# print(image)
# print(type(image))

# print(image.shape)
# print(type(image.shape))

# image_b[:, :, (1, 2)] = 0
# image_g[:, :, (0, 2)] = 0
# image_r[:, :, (0, 1)] = 0

# show_result("image blue", image_b)
# show_result("image green", image_g)
# show_result("image red", image_r)


# image_hstack = np.hstack((image_b, image_g, image_r))
# image_vstack = np.vstack((image_hstack, image_hstack, image_hstack))

# show_result('image_vstack', image_vstack)
# # show_result('image_hstack', image_hstack)


# END OF Pert 1
