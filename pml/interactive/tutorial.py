# Copyright (C) 2012 David Rusk
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to 
# deal in the Software without restriction, including without limitation the 
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
# sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS 
# IN THE SOFTWARE.
"""
An interactive tutorial of pml's basic functionality.

@author: drusk
"""

import platform
from code import InteractiveInterpreter
from subprocess import call

from pml.api import *

# Passing globals() to interpreter gives it access to the pml api imported 
# above.
tutorial_interpreter = InteractiveInterpreter(globals())

def get_tutorial_lessons():
    """
    Returns:
      lessons: dict
        Keys are the names of lessons, the value is the function which will 
        start the lesson.
    """
    return {"datasets": lesson_dataset_intro,
            "classifiers": lesson_classifiers,
            "decision_trees": lesson_decision_trees,
            "pca": lesson_pca}

def begin_tutorial():
    """
    Begin the tutorial.
    """
    clear_shell()
    print "+----------------------------+"
    print " Welcome to the PML tutorial."
    print "+----------------------------+"
    print "\n"
    print "This is an interactive tutorial that will teach you the basics "
    print "of using PML for data analysis and machine learning tasks."
    print "\n"
    
    wait_for_enter_to_continue()
    
    lesson_dataset_intro()
    
    end_tutorial()

def lesson_dataset_intro():
    """
    Introductory lesson on loading and examining data.
    """
    clear_shell()
    print "PML DataSet Introduction"
    print "------------------------"
    print "This lesson introduces the basics of PML's core data structure, the DataSet."
    print "A DataSet consists of a collection of samples which have values for a set of"
    print " features.  An example might be specific course grades for a set of students."
    print "\n"
    print "DataSets can be conveniently created by loading data from a file on disk using" 
    print "the 'load' function."
    print "\n"
    print "Try loading one of the tutorial data sets using the following command:"
    
    let_user_try("data = load(\"iris.csv\")")
    
    print "Correct.  The iris data set has now been loaded into memory."
    print "\n"
    print "By default the data is assumed to be in comma separated value format."
    print "Other formats can be loaded by specifying the \"delimiter\" option."
    print "For example, "
    print "    'data = load(\"iris.tsv\", delimiter=\"\\t\")'"
    print "would load a tab separated file."
    
    print "\n"
    wait_for_enter_to_continue()
    print "\n"
    
    print "Once a data set has been loaded, many operations can be performed on it."
    print "One of the simplest is listing its features.  Try it out:"
    
    let_user_try("data.feature_list()")
    
    print "Good.  You can also check the number of samples in the data set:"
    
    let_user_try("data.num_samples()")
    
    print "In this case we see there are 150 samples."
    print "\n"
    
    print "The iris data set that we loaded is labelled.  This means the actual "
    print "classifications of the samples are known.  You can check if a DataSet "
    print "is labelled like this:"
    
    let_user_try("data.is_labelled()")
    
    print "If you want a quick summary of the DataSet without calling all of these methods "
    print "individually, simply enter the name of the DataSet at the prompt:"
    
    let_user_try("data")
    
    print "To wrap up this lesson, lets try plotting our DataSet to see if there are "
    print "any visual patterns.  Since our DataSets can be more than two dimensional, "
    print "we need a visualization method such as 'Radviz' to fit it on a 2D plot."
    print "Try creating a plot (once it is created, close it to continue):"
    
    let_user_try("data.plot_radviz()")
    
    print "\n"
    print "That completes the introductory lesson on DataSets."
    print "The next lesson will cover using classifiers."
    
    if not should_quit():
        lesson_classifiers()

def lesson_classifiers():
    """
    Lesson on using classifiers.
    """
    clear_shell()
    print "Classifiers Lesson"
    print "------------------"
    print "This lesson teaches you how to create and use a classifier to predict"
    print "information about samples."
    
    print ""
    wait_for_enter_to_continue()
     
    print "\n"
    print "Classifiers are an example of 'supervised learning'.  They need to be "
    print "trained on example data before they can make predictions."
    print "Therefore, lets start by loading a data set:"
    
    let_user_try("data = load(\"iris.csv\")")
    
    print "Type the data set's name to see summary information about it:"
    
    let_user_try("data")
    
    print "We can see that this data set has 150 samples."
    print "We will use part of the data set as training examples, and part as test "
    print "data to evaluate the effectiveness of the classifier:"
    
    let_user_try("training, test = data.split(0.7, random=True)")
    
    print "We have just split 70% of the samples into a 'training' data set and the "
    print "remaining 30% into the 'test' data set.  Setting 'random=True' means the "
    print "samples in the training and test set can come from anywhere in the original"
    print "data set; they are not split sequentially.  This can give a better sampling "
    print "of training data if the original data set had some order to it."
    
    print ""
    wait_for_enter_to_continue()
    
    print "\n"
    print "We can verify the splitting has occurred correctly by checking the summary "
    print "information for each of the new data sets:"
    
    let_user_try("training")
    
    print "105 / 150 = 70% of the samples as expected."
    
    print "Now try:"
    let_user_try("test")
    
    print "45 / 150 = 30%, accounting for the rest of the samples."
    
    print ""
    wait_for_enter_to_continue()
    print "\n"
    
    print "Additionally, the original data set had 3 classifications of samples in it."
    print "The classifications were the species of the flower within the 'iris' genus."
    print "We can see the breakdown of species within the training set by using:"
    
    let_user_try("training.get_label_value_counts()")
    
    print "Note that since the original data was split randomly, the number of samples "
    print "in each class is also random.  The original data had a sequential 50-50-50 "
    print "split."

    print ""
    wait_for_enter_to_continue()

    print "\n"
    print "Now that we have our data, we can build a classifier."
    print "Selecting an appropriate classifier requires some knowledge about your data."
    print "Our data is all numerical measurements related to features of flowers."
    
    print "For numerical data, the K-Nearest Neighbours classifier (Knn for short) is a"
    print "good place to start.  It classifies new data by finding the 'k' most similar"
    print "samples in the training data and checking their classifications."
    
    print "\n"
    print "Lets create a Knn classifier with a k of 3."
    
    let_user_try("classifier = Knn(training, k=3)")
    
    print "Like with data sets, you can get some summary information about the "
    print "classifier by simply typing its name at the prompt."
    
    let_user_try("classifier")
    
    print "We will start by classifying a single sample.  Select the sample with id 1"
    print "from the original data set as follows:"
    
    let_user_try("sample = data.get_row(1)")
    
    print "Like with most objects in PML, you can see its properties by typing its "
    print "name at the prompt:"
    
    let_user_try("sample")
    
    print "Use the Knn classifier to predict this sample's species:"
    
    let_user_try("classifier.classify(sample)")
    
    print "The correct answer is 'Iris-setosa'"
    
    wait_for_enter_to_continue()
    
    print "\n"
    print "Next lets classify all of the samples in the 'test' data set."
    
    let_user_try("results = classifier.classify_all(test)")
    
    print "We can then check how well the classifier performed:"
    
    let_user_try("results.compute_accuracy()")
    
    print "\n"
    print "This represents the percentage of samples which were assigned the same"
    print "classification by the classifier as they were originally labelled with."
    print "We can only compute this accuracy if the actual labels are known.  You "
    print "will get a warning if you try it on an unlabelled data set."

    print ""
    wait_for_enter_to_continue()
    
    print "\n"
    print "That completes the introductory lesson on classifiers."
    print "The next lesson will cover decision trees."
    
    wait_for_enter_to_continue()
        
    if not should_quit():
        lesson_decision_trees()
    
def lesson_decision_trees():
    """
    Lesson on using decision trees.
    """
    clear_shell()
    print "Decision Trees Lesson"
    print "---------------------"
    print "Decision trees are another type of classifier which have the extra "
    print "advantage that their decision making model can be visualized to gain "
    print "further insight."
    
    print ""
    wait_for_enter_to_continue()
     
    print "\n"
    print "Decision trees are best used on data with discrete values.  Let's load a "
    print "sample data set:"
    
    let_user_try("data = load(\"play_tennis.csv\")")
    
    print "Check what features the data has:"
    
    let_user_try("data.feature_list()")
    
    print "The values for each of these values are discrete.  For example, possible "
    print "values for 'Temperature' are 'Hot', 'Mild' and 'Cool', while Outlook can "
    print "have values 'Sunny', 'Overcast' or 'Rain'."
    
    print ""
    wait_for_enter_to_continue()
    
    print "\n"
    print "You can check this information for yourself as follows:"
    
    let_user_try("data.get_feature_values(\"Temperature\")")
    
    print "We can build a decision tree as follows:"
    
    let_user_try("tree = DecisionTree(data)")
    
    print "To visualize the tree, use the plot method:"
    
    let_user_try("tree.plot()")
    
    print "We could see that 'Outlook' was at the root of the tree.  It is the most"
    print "important feature for the decision tree when it is classifying a sample."
    print "We can also see that temperature is never used as one of the decision nodes"
    print "so it doesn't seem to be as big a factor in whether tennis is played or not."
    
    print ""
    wait_for_enter_to_continue()
    
    print "The 'leaf' nodes at the bottom of the tree have values 'True' or 'False'"
    print "indicating whether or not the conditions are appropriate for playing tennis."
    
    print "\n"
    print "Let's try creating our own sample and then classifying it.  We can create a"
    print "new sample by hand as follows:"
    
    let_user_try("sample = {\"Outlook\": \"Sunny\", \"Temperature\": \"Cool\", \"Humidity\": \"Normal\", \"Wind\": \"High\"}")
    
    print "We can then classify the sample using our decision tree in the same way as"
    print "we used Knn in the previous lesson:"
    
    let_user_try("tree.classify(sample)")
    
    print "It looks like the conditions are suitable for tennis.  We can plot the tree"
    print "again and easily trace through it ourselves to see how this decision was "
    print "determined."
    
    let_user_try("tree.plot()")
    
    wait_for_enter_to_continue()
    
    print "\n"
    print "That completes the lesson on decision trees."
    print "The next lesson covers principal component analysis."
    
    wait_for_enter_to_continue()
    
    if not should_quit():
        lesson_pca()
    
def lesson_pca():
    """
    Lesson on principal component analysis.
    """
    clear_shell()
    print "PCA Lesson"
    print "----------"
    print "PCA stands for Principal Component Analysis."
    print "It is a means of reducing the size of your data while minimizing loss" 
    print "of useful information.  If you consider each feature to be a dimension"
    print "or axis, it does this by shifting to a new set of axes where the variance"
    print "in the first few axes is maximized, and the remainder become insignificant."
    
    print ""
    wait_for_enter_to_continue()
    print ""
    
    print "Lets load the iris data set and see how it can be reduced."
    
    let_user_try("data = load(\"iris.csv\")")
    
    print "We can see how much variance would be in each principal component of this"
    print "data set as follows:"
    
    let_user_try("get_pct_variance_per_principal_component(data)")
    
    print "In this case the first principal component has over 92% of the total "
    print "variance."
    
    print ""
    wait_for_enter_to_continue()
    print ""
    
    print "We can get a visual representation of this information using:"
    
    let_user_try("plot_pct_variance_per_principal_component(data)")
    
    print "If we know how much of the variance we want to keep, we can also have PML"
    print "recommend to us how many principal components should be kept.  Lets say "
    print "we want to keep 97%:"
    
    let_user_try("recommend_num_components(data, 0.97)")
    
    print "It tells us we only need 2 components, so by using PCA we can reduce this "
    print "data to 2 features instead of 4."
    
    let_user_try("reduced_data = pca(data, 2)")
    
    print "Lets check the summary information for this reduced data set:"
    
    let_user_try("reduced_data")
    
    print "You can see the original features are replaced with new ones named 0 and 1"
    print "which actually combine information from the previous features since the axes"
    print "have been shifted."
    
    print ""
    wait_for_enter_to_continue()
    print ""
    
    print "You can check the exact percent variance in this new data (with respect "
    print "to the original data):"
    
    let_user_try("reduced_data.percent_variance()")
    
    print "This reduced_data can then be used with classifiers like Knn with improved"
    print "performance since the number of features involved has been reduced."
    
    print ""
    wait_for_enter_to_continue()
    
    print "\n"
    print "That completes the lesson on PCA."
    
    wait_for_enter_to_continue()
    
    end_tutorial()

def end_tutorial():
    """
    Exit the tutorial.
    """
    clear_shell()
    print "-" * 70
    print "You have reached the end of the PML tutorial, congratulations!"
    print "You may also find the module documentation useful.  It can "
    print "be accessed from the PML shell by typing 'docs'."
    print "-" * 70


########################## Utilities ###########################

def let_user_try(command):
    """
    Lets the user try entering the command.
    
    Args:
      command: string
        The command the user should enter.
    """
    print "\n"
    
    while True:
        print command
        print "\n"
        
        user_command = pml_prompt()
        
        if commands_match(user_command, command):
            tutorial_interpreter.runsource(user_command)
            print "\n"
            return
        
        print "\n"
        print "The command was not entered correctly, try again."
        
def commands_match(user_command, expected_command):
    """
    Checks if the commands are essentially equivalent after whitespace
    differences and quote differences.
    
    Returns:
      match: boolean
        True if the commands appear to be equivalent.
    """
    def normalize(command):
        return command.replace(" ", "").replace("'", "\"")
    
    return normalize(user_command) == normalize(expected_command)

def wait_for_enter_to_continue():
    """
    Prompt the user to press enter before continuing.
    """
    raw_input("[Press enter to continue]")

def pml_prompt():
    """
    Reads user input given the PML prompt.
    """
    return raw_input("pml:> ")

def clear_shell():
    """
    Clears the shell's screen.
    """
    if platform.system() == "Windows":
        # shell=True for reason in this stackoverflow answer
        # http://stackoverflow.com/questions/3022013/windows-cant-find-the-file-on-subprocess-call
        call("cls", shell=True)
    else:
        call("clear")

def should_quit():
    """
    Asks the user if they want to continue with the next lesson or quit the 
    tutorial.
    
    Returns:
      should_quit: boolean
        True if the user wants to quit the tutorial.
    """
    print "\n"
    print ("Type 'quit' to end the tutorial, or press enter to continue to "
           "the next lesson.")
    selection = pml_prompt()
    return selection.lower() == "quit"
