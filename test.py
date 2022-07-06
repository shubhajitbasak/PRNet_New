import numpy as np


def test(self, error_func_list=None, is_visualize=False):
    from demorender import demoAll
    total_task = len(self.test_data)
    print('total img:', total_task)

    model = self.net.model
    total_error_list = []
    num_output = self.mode[3]
    num_input = self.mode[4]
    data_generator = DataGenerator(all_image_data=self.test_data, mode=self.mode[2], is_aug=False, is_pre_read=self.is_pre_read)

    with torch.no_grad():
        model.eval()
        for i in range(len(self.test_data)):
            data = data_generator.__getitem__(i)
            x = data[0]
            x = x.to(self.net.device).float()
            y = [data[j] for j in range(1, 1 + num_input)]
            for j in range(num_input):
                y[j] = y[j].to(x.device).float()
                y[j] = torch.unsqueeze(y[j], 0)
            x = torch.unsqueeze(x, 0)
            outputs = model(x, *y)

            p = outputs[-1]
            x = x.squeeze().cpu().numpy().transpose(1, 2, 0)
            p = p.squeeze().cpu().numpy().transpose(1, 2, 0) * 280
            b = sio.loadmat(self.test_data[i].bbox_info_path)
            gt_y = y[0]
            gt_y = gt_y.squeeze().cpu().numpy().transpose(1, 2, 0) * 280

            temp_errors = []
            for error_func_name in error_func_list:
                error_func = getErrorFunction(error_func_name)
                error = error_func(gt_y, p, b['Bbox'], b['Kpt'])
                temp_errors.append(error)
            total_error_list.append(temp_errors)
            print(self.test_data[i].init_image_path, end='  ')
            for er in temp_errors:
                print('%.5f' % er, end=' ')
            print('')
            if is_visualize:

                if temp_errors[0] > 0.00:
                    tex = np.load(self.test_data[i].texture_path.replace('zeroz2', 'full')).astype(np.float32)
                    init_image = np.load(self.test_data[i].cropped_image_path).astype(np.float32) / 255.0
                    show([p, tex, init_image], mode='uvmap')
                    init_image = np.load(self.test_data[i].cropped_image_path).astype(np.float32) / 255.0
                    show([gt_y, tex, init_image], mode='uvmap')
                    demobg = np.load(self.test_data[i].cropped_image_path).astype(np.float32)
                    init_image = demobg / 255.0
                    img1, img2 = demoAll(p, demobg, is_render=False)
            mean_errors = np.mean(total_error_list, axis=0)
            for er in mean_errors:
                print('%.5f' % er, end=' ')
            print('')
        for i in range(len(error_func_list)):
            print(error_func_list[i], mean_errors[i])

        se_idx = np.argsort(np.sum(total_error_list, axis=-1))
        se_data_list = np.array(self.test_data)[se_idx]
        se_path_list = [a.cropped_image_path for a in se_data_list]
        sep = '\n'
        fout = open('errororder.txt', 'w', encoding='utf-8')
        fout.write(sep.join(se_path_list))
        fout.close()


if __name__ == '__main__':
    t = np.load('data/uv-data/canonical_vertices.npy')
    print(t.shape)