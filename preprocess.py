import numpy as np

#ランダムノイズを加える関数
#percentage:ランダムな値にする割合
def make_random(data,percentage):
    threshold = percentage/100
    w,h = data.shape
    mask = np.random.rand(w,h) < threshold
    data += (np.random.rand(w,h)-data)*mask
    return data


#データオグメンテーション
#x shape: batch*size*size
def data_segmentation(x,theta_range,seg_type=None):
    shape = x.shape
    input_w = shape[2]
    input_h = shape[1]
    num = shape[0]
    
    #出力行列
    y = np.zeros((num,input_w,input_h))
    #データを回転させたものでかさ増し
    if seg_type == 'rotate':
        for i in range(num):
            #thetaをランダムで決定
            theta = np.random.uniform(theta_range[0], theta_range[1])
            #radianに変換
            rad_theta = np.radians(theta)
            y[i,:,:] = rotate(x[i,:,:],rad_theta)
    return y

#画像を回転させたものを返す
def rotate(input_image,theta):
    # theta:radian
    #image size: 28*28
    image_shape = input_image.shape
    #画像の縦横
    src_w = image_shape[1]
    src_h = image_shape[0]
    pad = 6 #データがはみ出さないように設定
    # 画像のパディング（28x28の画像を(28+pad x 28+pad）の画像になるように周辺を白埋め）
    image = np.zeros((src_h + pad, src_w + pad))
    image[int(pad/2):int(pad/2)+src_h, int(pad/2):int(pad/2)+src_w] = input_image
    
    # 平行移動して中心を(0, 0)にする
    par_array1 = np.asarray([[1, 0, -src_w/2], [0, 1, -src_h/2], [0, 0, 1]]).astype(float)
    # 回転
    rot_array = np.asarray([[np.cos(theta), np.sin(theta), 0],[-np.sin(theta), np.cos(theta), 0],[0, 0, 1]],dtype=np.float).astype(float)
    # 平行移動して中心を戻す
    par_array2 = np.asarray([[1, 0, src_w/2], [0, 1, src_h/2], [0, 0, 1]]).astype(float)

    # アフィン行列を計算
    affine_array = np.dot(par_array2,np.dot(rot_array, par_array1))
    # 逆行列を求める
    inv_affine_array = np.linalg.inv(affine_array)
    # 出力画像のピクセル位置のarray
    out_array = np.asarray([[out_x, out_y, 1] for out_x in range(src_w) for out_y in range(src_h)]).T
    # 各出力位置に対応した入力画像の位置のarray
    src_array = np.dot(inv_affine_array, out_array)
    
    #Bilinear補完
    # まず，各出力画素に対応する入力画素をx, yごとに (784, )のshapeのarrayにまとめる
    # → src_xのi番目の要素は，出力画像の(int(i/28), i%28)に位置する画素に対応する入力画素のx座標を表す
    src_x = src_array[0, :].T
    src_y = src_array[1, :].T

    #各入力画素の最近傍に位置する画素（座標の値が整数)
    floor_src_x = np.floor(src_x).astype(int)
    floor_src_y = np.floor(src_y).astype(int)
    
    #最近傍画素の周辺画素の座標のリスト
    x0 = np.clip(floor_src_x, 0, src_w - 1)
    x1 = np.clip(floor_src_x + 1, 0, src_w - 1)
    y0 = np.clip(floor_src_y, 0, src_h - 1)
    y1 = np.clip(floor_src_y + 1, 0, src_h - 1)

    # 周辺画素の画素値を取得
    src_a = input_image[y0, x0]
    src_b = input_image[y1, x0]
    src_c = input_image[y0, x1]
    src_d = input_image[y1, x1]

    # 周辺画素との距離を計算する
    dx = src_x - floor_src_x
    dy = src_y - floor_src_y
    
    # 平均する際にかける重みを計算
    weight_a = (1 - dx) * (1 - dy)
    weight_b = (1 - dx) * dy
    weight_c = dx * (1 - dy)
    weight_d = dx * dy
    
    # 重み付き平均の計算
    output = (weight_a * src_a + weight_b * src_b + weight_c * src_c + weight_d * src_d).reshape(image_shape).T
    
    return output.astype(int)
