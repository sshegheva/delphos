import glob
import logging
import pandas as pd
import traceback
FEATURES = []

PLACE = 'place'
INNO_DAY = 'innovation_day'
DEPARTMENT = 'department'
LOCATION = 'location'
DIVERSITY = 'diversity'
TEAM_MODE = 'team_mode'
TEAM_SIZE = 'team_size'
TEAM_MEMBERS = 'members'
TOPIC = 'topic'
DEMO = 'demo'
STYLE = 'style'
CATEGORY = 'category'
PRESENTATION_RANK = 'rank'
GENDER_DIVERSE = 'gender_diverse'
TEAM_NAME = 'team_name'
TEAM_NAME_LENGTH = 'team_name_length'
PRESENTATION_IMAGE = 'image'
JUDGES = 'judges'
EASTER_EGG = 'easter_egg'

"""
raw extracted features

['Place',
 'Team Name',
 'Project Home Base (NYC, Bos, Germany, etc.)',
 'Classic or GSD?',
 'Session',
 'Quick Description of the project (for recruiting others!)',
 'Team Members (no more than 5!)']

"""

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


class FeatureExtractor(object):
    def __init__(self, innovation_csv, debug=False):
        LOGGER.debug('Parsing {}'.format(innovation_csv))
        self.df = None
        try:
            self.csv = innovation_csv
            self.name = self.csv.split('/')[-1].split('.')[0]
            self.df = pd.read_csv(self.csv)
            self.df.rename(columns=lambda x: x.strip(), inplace=True)
            self.columns = self.df.columns.tolist()
            self._parse_features()
            self._drop_noise()
            LOGGER.debug('Parsed {}'.format(innovation_csv))
        except:
            if debug:
                traceback.print_exc()
            LOGGER.error('Unable to parse {}'.format(innovation_csv))

    def _parse_features(self):
        self.create_category_feature()
        self.create_department_feature()
        self.create_diversity_feature()
        self.create_easter_egg_feature()
        self.create_gender_diversity_feature()
        self.create_is_demo_feature()
        self.create_judges_feature()
        self.create_location_feature()
        self.create_place_feature()
        self.create_presentation_image_feature()
        self.create_team_members_feature()
        self.create_team_mode_feature()
        self.create_team_name_feature()
        self.create_team_size_feature()
        self.create_topic_feature()
        self.create_style_feature()
        self.create_innovation_day_feature()

    def _drop_column(self, col_name):
        self.df.drop(col_name, axis=1, inplace=True)

    def _drop_noise(self):
        for col in self.df.columns:
            if 'Unnamed' in col:
                self._drop_column(col)

    def create_place_feature(self):
        if 'Place' in self.df.columns.tolist():
            self.df[PLACE] = self.df['Place']
            self._drop_column('Place')
        else:
            self.df[PLACE] = 'TBD'

    def create_innovation_day_feature(self):
        self.df[INNO_DAY] = self.name

    def create_topic_feature(self):
        columns = self.df.columns.tolist()
        column = None
        for c in columns:
            if 'Description' in c:
                column = c
        if column:
            self.df[TOPIC] = self.df[column]
            self._drop_column(column)

    def create_department_feature(self):
        pass

    def create_location_feature(self):
        columns = self.df.columns.tolist()
        column = None
        for c in columns:
            if 'Home Base' in c:
                column = c
        if column:
            self.df[LOCATION] = self.df[column]
            self._drop_column(column)

    def create_team_members_feature(self):
        columns = self.df.columns.tolist()
        column = None
        for c in columns:
            if 'Team Members' in c:
                column = c
        if column:
            self.df[TEAM_MEMBERS] = self.df[column].apply(lambda x: filter(None, str(x).replace('\t', '\n').splitlines()))
            self.df[TEAM_MEMBERS] = self.df[TEAM_MEMBERS].apply(lambda x: [i for i in x if i != '' and i != '\xe2\x80\xa2'])
            self._drop_column(column)

    def create_diversity_feature(self):
        pass

    def create_team_mode_feature(self):
        pass

    def create_team_size_feature(self):
        self.df[TEAM_SIZE] = self.df[TEAM_MEMBERS].apply(lambda x: len(x))

    def create_is_demo_feature(self):
        pass

    def create_style_feature(self):
        if 'Classic or GSD?' in self.columns:
            self.df[STYLE] = self.df['Classic or GSD?']
            self._drop_column('Classic or GSD?')
        else:
            self.df[STYLE] = 'Classic'

    def create_category_feature(self):
        pass

    def create_presentation_rank_feature(self):
        pass

    def create_gender_diversity_feature(self):
        pass

    def create_team_name_feature(self):
        self.df[TEAM_NAME] = self.df['Team Name']
        self._drop_column('Team Name')

    def create_team_name_length(self):
        self.df[TEAM_NAME_LENGTH] = self.df[TEAM_NAME].apply(lambda x: sum(x))

    def create_presentation_image_feature(self):
        pass

    def create_easter_egg_feature(self):
        pass

    def create_judges_feature(self):
        pass


def create_innovation_frame(data_path):
    data_files = glob.glob(data_path + '/*.csv')
    frames = [FeatureExtractor(f).df for f in data_files]
    return pd.concat(frames)
