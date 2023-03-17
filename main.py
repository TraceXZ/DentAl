from pages.home import MainPage
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-top_low', type=float, default=0.05)
    parser.add_argument('-top_high', type=float, default=0.95)


    args = parser.parse_args()
    mainpage = MainPage(args)

