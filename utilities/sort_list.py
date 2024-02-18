'''takes a python list and prints an alphabetically sorted version'''


user_favorite_movies = [
    "Ender's Game",
    'Lord of the Rings',
    'The Hobbit',
    'The Silmarillion',
    "Hitchhiker's Guide to the Galaxy",
    'Life, the Universe and Everything',
    'The Color of Magic',
    'The Four Agreements',
    'The Redemption of Althalus',
    'Harry Potter',
    'Emotional Intelligence 2.0',
    'The Book of Five Rings',
    'The Chronicles of Narnia',
    'The Screwtape Letters',
    'Dune',
    'Angels & Demons',
    '1984',
    'Animal Farm',
    'The Dark Tower series',
    'The Call of Cthulhu',
    'Good Omens'
]

list_name = user_favorite_movies

list_name.sort()

print('# Sorted')
print(f'sorted_list = [')
for word in list_name[:-1]:
    print(f'    "{word}",')
print(f'    "{list_name[-1]}"')
print(']')