using DeepLearn.NeuralNetwork.Base.Entities;

namespace NeuralNet.Entities
{
    public class NewSynapse : Synapse
    {
        public NewSynapse(int sourceIndex, int destinationIndex) 
            : base(sourceIndex, destinationIndex)
        {
            Prune = false;
        }

        public NewSynapse(int sourceIndex, int destinationIndex, double weight)
            : base(sourceIndex, destinationIndex, weight)
        {
            Prune = false;
        }

        public bool Prune { get; set; }

        public override object Clone()
        {
            var synapseCopy = new NewSynapse(this.IndexSource, this.IndexDestination)
            {
                Weight = this.Weight,
                Signal = this.Signal,
                Prune = this.Prune
            };

            return synapseCopy;
        }
    }
}
