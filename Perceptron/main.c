#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <math.h>

int main()
{
	time_t t;
	srand((unsigned) time(&t));
	int i;
	float xweight=(float)(rand()%1000)/1000;
	float yweight=(float)(rand()%1000)/1000;
	float bias=(float)(rand()%1000)/1000;
	float alpha=2;
	float correct=0;
	for(i=0;i<100000;i++)
	{
		float x=(float)(rand()%1000)/1000;
		float y=(float)(rand()%1000)/1000;
		float desired_output=0.0;
		if(x*2+0.1>y)
			desired_output=1.0;
		float activation=1/(1+exp(-(x*xweight+y*yweight+bias)));
		float output;
		if(activation>0.5)
			output=1.0;
		else
			output=0.0;
		if(output==desired_output)
			correct++;
		printf("x=%.3f %s \n", x, (output==desired_output) ? "CORRECT" : "WRONG" );
		// printf(" weight:%.3f bias %.3f\n", weight, bias);
		float error=desired_output-activation;
		float dirac = activation*(1-activation)*error;
		float dx_weight=alpha*x*dirac;
		float d_bias=alpha*1*dirac;
		float dy_weight=alpha*y*dirac;
		// printf(" error:%.3f\n", error);
		// printf(" dweight:%.3f dbias %.3f\n", d_weight, d_bias);
		xweight+=dx_weight;
		bias+=d_bias;
		yweight+=dy_weight;
		// char c;
		// scanf("%c",&c);
	}
	printf("Correct:%f\n",correct/100000);
}