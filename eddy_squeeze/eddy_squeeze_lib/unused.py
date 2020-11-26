
    def get_basic_diff_info(self):
        self.df.groupby(
            ['number of volumes',
             'max b value',
             'min b value',
             'number of b0s']).count()['subject'].to_frame()
