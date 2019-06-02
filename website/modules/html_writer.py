import json
import os
import os.path
import string
from sklearn.utils import check_random_state


def id_generator(size=15, random_state=None):
    """Helper function to generate random div ids. This is useful for embedding
    HTML into ipython notebooks."""
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(random_state.choice(chars, size, replace=True))


def save_to_file(exp,
                 labels=None,
                 predict_proba=True,
                 show_predicted_value=True,
                 **kwargs):
    """Saves html explanation to file. .

    Params:
        file_path: file to save explanations to

    See as_html() for additional parameters.

    """
    return as_html(exp, labels=labels,
                            predict_proba=predict_proba,
                            show_predicted_value=show_predicted_value,
                            **kwargs)
    # file_ = open(file_path, 'w', encoding='utf8')
    # file_.write(exp.as_html(labels=labels,
    #                         predict_proba=predict_proba,
    #                         show_predicted_value=show_predicted_value,
    #                         **kwargs))
    # file_.close()


def as_html(explainer,
            labels=None,
            predict_proba=True,
            show_predicted_value=True,
            **kwargs):
    """Returns the explanation as an html page.

    Args:
        labels: desired labels to show explanations for (as barcharts).
            If you ask for a label for which an explanation wasn't
            computed, will throw an exception. If None, will show
            explanations for all available labels. (only used for classification)
        predict_proba: if true, add  barchart with prediction probabilities
            for the top classes. (only used for classification)
        show_predicted_value: if true, add  barchart with expected value
            (only used for regression)
        kwargs: keyword arguments, passed to domain_mapper

    Returns:
        code for an html page, including javascript includes.
    """

    def jsonize(x):
        return json.dumps(x, ensure_ascii=False)

    if labels is None and explainer.mode == "classification":
        labels = explainer.available_labels()

    this_dir, _ = os.path.split(__file__)
    # bundle = open(os.path.join(this_dir, 'bundle.js'),
    #               encoding="utf8").read()
    #
    # bundle_js = '''%s''' % bundle
    out = ''''''
    # random_id = id_generator(size=15, random_state=check_random_state(explainer.random_state))
    # out += u'''
    # <div class="lime top_div" id="top_div%s"></div>
    # ''' % random_id

    predict_proba_js = ''
    if explainer.mode == "classification" and predict_proba:
        predict_proba_js = '''
        var pp_div = proba_div.append('div')
                            .classed('lime predict_proba', true);
        var pp_svg = pp_div.append('svg').style('width', '100%%');
        '''
        # var pp = new lime.PredictProba(pp_svg, %s, %s);
        #% (jsonize([str(x) for x in explainer.class_names]),
            #   jsonize(list(explainer.predict_proba.astype(float))))

    # predict_value_js = ''
    # if explainer.mode == "regression" and show_predicted_value:
    #     # reference self.predicted_value
    #     # (svg, predicted_value, min_value, max_value)
    #     predict_value_js = u'''
    #             var pp_div = proba_div.append('div')
    #                                 .classed('lime predicted_value', true);
    #             var pp_svg = pp_div.append('svg').style('width', '100%%');
    #             var pp = new lime.PredictedValue(pp_svg, %s, %s, %s);
    #             ''' % (jsonize(float(explainer.predicted_value)),
    #                    jsonize(float(explainer.min_value)),
    #                    jsonize(float(explainer.max_value)))

    exp_js = '''var exp_div;
        var exp = new lime.Explanation(%s);
    ''' % (jsonize([str(x) for x in explainer.class_names]))

    if explainer.mode == "classification":
        for label in labels:
            exp = jsonize(explainer.as_list(label))
            exp_js += '''
            exp_div = exp_div.append('div').classed('lime explanation', true);
            '''
            #exp.show(%s, %d, exp_div); % (exp, label)
    else:
        exp = jsonize(explainer.as_list())
        exp_js += '''
        exp_div = exp_div.append('div').classed('lime explanation', true);
        '''
        # exp.show( % s, % s, exp_div); % (exp, explainer.dummy_label)

    raw_js = '''var raw_sub_div = raw_div.append('div');'''

    if explainer.mode == "classification":
        html_data = explainer.local_exp[labels[0]]
    else:
        html_data = explainer.local_exp[explainer.dummy_label]

    raw_js += explainer.domain_mapper.visualize_instance_html(
        html_data,
        labels[0] if explainer.mode == "classification" else explainer.dummy_label,
        'raw_sub_div',
        'exp',
        **kwargs)

    out += '''
    var proba_div = d3.select('#figure1').classed('lime top_div', true);
    var exp_div = d3.select('#figure2').classed('lime top_div', true);
    var raw_div = d3.select('#figure3').classed('lime top_div', true);
    %s
    %s
    %s
    ''' % (predict_proba_js, exp_js, raw_js)
    # out += u'</body></html>'

    return out