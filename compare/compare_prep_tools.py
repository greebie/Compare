from os import listdir
import pickle

class prep_tools:
    """ prep_tools prepares outputs from warcbase for use in Compare
        
        
        """
    
    def __init__(self, filepath):
        self.TRUNCATE_LENGTH = 15
        self.filepath = filepath
        self.dummies = ()
    
    def text_tuple_to_tuple (self, *args):
        return (tuple(map (str.strip, line.translate(str.maketrans('', '', '()')).split())))

    def processurl (self, subdomain=False):
        """ takes a warcbase url output and prepares it for Compare.
            
            """
        urls = []
        for filename in os.listdir(self.filepath):
                with open(self.filepath+filename, "r") as file:
                    if subdomain:
                        urls.append(list({(filename[0:self.TRUNCATE_LENGTH],
                                       text_tuple_to_tuple(line)[0]][2:6],
                                       text_tuple_to_tuple(line)[1]) for line in file.readlines()}))
                    else:
                        urls.append(list({(filename[0:self.TRUNCATE_LENGTH],
                                       text_tuple_to_tuple(line)[0]][2:6],
                                       '.'.join(text_tuple_to_tuple(line)[1].split('.')[-2:0]) for line in file.readlines()}))
        return(urls)

    def test_dummies (self, dummies=()):
        """ All dummies must contain unique values or will influence analysis """
        return len(set(sum(dummies, ()))) == len(sum(dummies, ()))
    
    def filter (self, PC):
        """ Remove dummies and exclude collections from analysis. """
        d = dict()
        for collect in PC:
            for coll, date, url in collect:
                if coll not in self.dummies or coll not in self.exclude:
                        d[date][coll].append(url)
        return (d)

    def add_stop_words (self, words=[]):
        try:
            with open(self.session_filename, "rb") as f:
                old_words = Pickle.loads("stop_words.p", rw)
                pickle.dump(old_words.update(words), open("stop_words.p", wb))
        except pickle.UnpicklingError as e:
                # normal, somewhat expected
                continue
        except (AttributeError,  EOFError, ImportError, IndexError) as e:
                # secondary errors
                print(traceback.format_exc(e))
                continue
        except Exception as e:
                # everything else, possibly fatal
                print(traceback.format_exc(e))
                return

