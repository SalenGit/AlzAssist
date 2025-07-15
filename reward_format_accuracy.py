from swift.plugin import ORM, orms


class ADFormatAccuracyFunction(ORM):
        
    def __call__(self, completions, **kwargs):


        
        all_labels = kwargs.get('auxiliaries')

        endwith_dic = {
            0: ('Assessment Classification: Healthy_Control', 'Assessment Classification: Healthy_Control.', 'Assessment Classification:Healthy_Control', 'Assessment Classification:Healthy_Control.'),
            1: ('Assessment Classification: Mild_Cognitive_Impairment', 'Assessment Classification: Mild_Cognitive_Impairment.', 'Assessment Classification:Mild_Cognitive_Impairment', 'Assessment Classification:Mild_Cognitive_Impairment.'),
            2: ('Assessment Classification: Alzheimers_Disease', 'Assessment Classification: Alzheimers_Disease.', 'Assessment Classification:Alzheimers_Disease', 'Assessment Classification:Alzheimers_Disease.'),
        }

        
        endwith_all = ('Assessment Classification: Healthy_Control', 'Assessment Classification: Healthy_Control.', 'Assessment Classification:Healthy_Control', 'Assessment Classification:Healthy_Control.', 'Assessment Classification: Mild_Cognitive_Impairment', 'Assessment Classification: Mild_Cognitive_Impairment.', 'Assessment Classification:Mild_Cognitive_Impairment', 'Assessment Classification:Mild_Cognitive_Impairment.', 'Assessment Classification: Alzheimers_Disease', 'Assessment Classification: Alzheimers_Disease.', 'Assessment Classification:Alzheimers_Disease', 'Assessment Classification:Alzheimers_Disease.')

        all_scores = []
        for pred, label in zip(completions, all_labels):
            
            endwith_true = endwith_dic[label]
            
            if pred.endswith(endwith_true):
                all_scores.append(1.0)
            elif pred.endswith(endwith_all):
                all_scores.append(0.5)
            else:
                all_scores.append(0.0)
                
        return all_scores


orms['format_accuracy']= ADFormatAccuracyFunction