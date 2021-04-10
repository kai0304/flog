import imgaug as ia
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt
import imageio
import glob
import os.path

def remove_glob(pathname):
    for p in glob.glob(pathname, recursive=False):
        if os.path.isfile(p):
            os.remove(p)

#加工後画像を保存
def writeFile(filepath, dir_out, prefix, i, aug_img):
    filename = os.path.basename(filepath)
    root, ext =  os.path.splitext(filename)
    outpath = dir_out + '/' + root + '_' + prefix + '_' +str(i) + ext
    imageio.imwrite(outpath,aug_img)
    print('out: '+ outpath)

#変換器

#ノイズを入れる
def noise(filepath, dir_out, img, params):
    i = 0
    for d in params:
        i = i + 1

        augDropout = iaa.Dropout(p=d)
        aug_img = augDropout.augment_image(img)
        writeFile(filepath, dir_out + '/noise', 'noise', i, aug_img)



#色反転　反転確率　画像の割合
def invert(filepath, dir_out, img, rate, per_channel):
    i = 1
    augInvert = iaa.Invert(rate, per_channel)
    aug_img = augInvert.augment_image(img)
    writeFile(filepath, dir_out + '/invert', 'invert', i, aug_img)

#エッジ検出
def edge(filepath, dir_out, img, alpha_under, alpha_top):
    i = 1
    augEdge = iaa.EdgeDetect(alpha=(alpha_under, alpha_top))
    aug_img = augEdge.augment_image(img)
    writeFile(filepath, dir_out + '/edge', 'edge', i, aug_img)



def main(dir_in, dir_out):
    #出力先のディレクトリをクリーン
    remove_glob(dir_out + '/*')
    #実行ディレクトリ表示
    
    for filepath in glob.glob(dir_in + '/*'):
        print('in: ' + filepath)
        #imgに画像を読み込む
        img = imageio.imread(filepath)
        #変換器
        #ノイズ　リスト複数入力可
         noise(filepath, dir_out, img, [0.3])
        #色反転
        invert(filepath, dir_out, img, 0.3, 1.0)
        #エッジ検出
        edge(filepath, dir_out, img, 0.3, 0.6)
                
dir = os.path.dirname(os.path.abspath(__file__))
print(dir)
main(dir + '/data/original/cytospin', dir + '/data/processing')
