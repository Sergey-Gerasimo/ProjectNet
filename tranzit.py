from PIL import Image 
import sys, os

from cv2 import resizeWindow 

success = True 
list_train = os.listdir('train') # список подкатоалогов train
list_val = os.listdir('val')
list_test = os.listdir('test')

FOLDERS = {'test':'tmp_test', 
            'train':'tmp_train', 
            'val':'tmp_val'}

# создаем директории
for i in FOLDERS:
    os.mkdir(FOLDERS[i])

def write(count: int, step: int) -> None:
    "печатет строку процентов"
    LINE = 50 # длина строки 
    proc = (step*100)//count 
    procline = (step*LINE)//count

    str = f'{proc:>4}% ' +'|' + '='*procline + '>' + ' '*(LINE - procline-1) + f'| {step}' + ' ' * 10
    sys.stdout.write('\r'+str)
    sys.stdout.flush()

def get_count_files(list_dir:list, name_folder:str) -> int:
    """"
    Возвращает количество файлов в каталоге с древовидной структурой с количеством вложений 2
    """

    count = 0
    try:
        for path in list_dir:
            p = os.getcwd() + '\\'+ name_folder + '\\' + path 
            count += len(os.listdir(p))
        else:
            return count

    except:
        return len(list_dir)

def resize(list_img: list, folder_name_from:str, folder_name_to:str) -> None:
    """
    перемещает и масштабирует изображения из folder_name_from в folder_name_to
    list_img: list -- список изображений в папке 
    """
    j = 1
    l = len(list_img)
    for i in list_img:
        path = os.getcwd() + '\\' + folder_name_from + '\\' + i
        try:
            img = Image.open(path)
            img = img.resize((256, 256))

            img.save(os.getcwd() + '\\' + folder_name_to + '\\' + f'{j}.JPEG')
            os.remove(path)
            j += 1 
            write(l, j)
        except:
            pass 

def normalize(dir_list: list, name_folder_from:str, name_folder_to:str) -> None:
    """
    Просматривает все подкоталоги из dir_list и перемещает в folder_name_to все изображения и масштабирует их 
    """
    i = 1
    count = get_count_files(dir_list, name_folder_from)

    for path in dir_list:
        p = os.getcwd() + '\\' + name_folder_from + '\\' + path # путь до папки 

        for image in os.listdir(p):
            try:
                img_t = p + '\\' + image # путь до картинки 
                img = Image.open(img_t)
                img = img.resize((256, 256))

                img.save(os.getcwd() + '\\' + name_folder_to + '\\' + f'{i}.JPEG')
                os.remove(img_t)
                write(count, i)
                i += 1
            except: pass 
        else:
            os.rmdir(p)

print('sample prparation')
print(' train:')
try:
    normalize(list_train, 'train', FOLDERS['train'])
    print('success')
except:
    success = False
    print('failed')

print(' test:')
try:
    resize(list_test, 'test', FOLDERS['test'])
    print('success')
except:
    success = False
    print('failed')

print(' val:')
try:
    resize(list_val, 'val', FOLDERS['val'])
    print('success')
except:
    success = False
    print('failed')
