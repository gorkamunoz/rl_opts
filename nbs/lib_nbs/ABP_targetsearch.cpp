//compile with: g++ ABP_targetsearch.cpp nrutil.cpp -o PS -O2 -w
// working if at each episode the glow matrix is initialized to zero
#include <stdio.h> 
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <string>
#include <sstream>
#include <fstream>
#include <bits/stdc++.h>
using namespace std;

#include "ABP_targetsearch.h"
#include "nrutil.h"

#define PI 3.14159265358979323846264338

 
int main () {
	// for parallelization and restarting (when restarting a parallel simulation, N_runs should not change)
	k_bunch = 0;
	N_episodes0 = 0;
	
	// reading various parameters
	read_parameters ();
	
	// environment parameters
	Nphi = 2;				// number of phases
	Ns = 20000;				// number of states
	Na = 2;					// number of actions
	
	// other settings
	srand (12345+k_bunch*N_episodes+k_bunch*N_episodes+N_episodes0);									// initialize random number generator
	
	// arrays initialization
	array_initialization();
	
	// utilities initialization
	v_factor1 = dvector(0,Nt_single_episode-1);
	v_factor2 = dvector(0,Nt_single_episode-1);
	H2 = dvector(0,Na-1);
	visited_states = ivector(0,Nt_single_episode-1);
	visited_actions = ivector(0,Nt_single_episode-1);
	v_factor1[0]=1.0;
	for (int i=1; i<Nt_single_episode; i++){
		v_factor1[i]=pow((1.0-gammma),i*1.0);
	}
	v_factor2[0] = 0.0;
	v_factor2[1] = gammma;
	for (int i=2; i<Nt_single_episode; i++){
		v_factor2[i] = v_factor2[i-1] + gammma*v_factor1[i-1];
	}

	////////////////////////////////////////////////////////////////////
	// run learning for several independent processes
	for (int run=0; run<N_runs; run++){
		printf("  %d out of %d \n" , run+1,N_runs);

		ProjectiveSimulationOptimized(run);

		for (int s=0; s<Ns; s++){
			Pvalues[run][s] = H[s][1]/(H[s][0]+H[s][1]);
			Svalues[run][s] = H[s][0];
			avg_Pvalues[s] += Pvalues[run][s];
			for (int a=0; a<Na; a++){
				avg_state_action_values[s][a]+=H[s][a];
				avg_glow[s][a]+=G[s][a];
			}
		}
	}

	// compute final averages
	for (int s = 0; s<Ns; s++){
		avg_Pvalues[s] = avg_Pvalues[s]/N_runs;
		for (int a=0; a<Na; a++){
			avg_state_action_values[s][a] = avg_state_action_values[s][a]/N_runs;
			avg_glow[s][a] = avg_glow[s][a]/N_runs;
		}
	}
	
	for (int episode=0; episode<N_episodes; episode++){
		for (int run=0; run<N_runs; run++){
			avg_target_times[episode]+=target_times[run][episode];
		}
		avg_target_times[episode]=avg_target_times[episode]/N_runs;
		for (int s = 0; s<Ns; s++){
			avg_Pvalues_episode[episode][s]=avg_Pvalues_episode[episode][s]/N_runs;
			avg_H0values_episode[episode][s]=avg_H0values_episode[episode][s]/N_runs;
			avg_H1values_episode[episode][s]=avg_H1values_episode[episode][s]/N_runs;
		}
	}

	// saving results
	print_averages ();
	print_data ();
}

////////////////////////////////////////////////////////////////////////
void read_parameters () {
	FILE* fp = fopen("Parameters.txt", "r");
	if (fp == NULL) {
		exit(EXIT_FAILURE);
	}
	char str [20];
	char* line = NULL;
	size_t len = 0;
	int k = -1;
	while ((getline(&line, &len, fp)) != -1) {
		// using printf() in all tests for consistency
		k++;
		printf("%d %s", k,line);
		
		if (k==1) {
			sscanf (line,"%s %*s %lf",str,&L);
			//~ printf("        L = %11.8f\n", L);
		} else if (k==2) {
			sscanf (line,"%s %*s %lf",str,&tau);
			//~ printf("        tau = %11.8f\n", tau);
		} else if (k==4) {
			sscanf (line,"%s %*s %lf",str,&Pe);
			//~ printf("        Pe = %11.8f\n", Pe);
		} else if (k==5) {
			sscanf (line,"%s %*s %lf",str,&ell);
			//~ printf("        ell = %11.8f\n", ell);
		} else if (k==6) {
			sscanf (line,"%s %*s %lf",str,&R_target);
			//~ printf("        R = %11.8f\n", R_target);
		} else if (k==8) {
			sscanf (line,"%s %*s %lf",str,&dt);
			//~ printf("        dt = %11.8f\n", dt);
		} else if (k==10) {
			sscanf (line,"%s %s %*s %d",str,str,&N_runs);
			//~ printf("        N runs = %d\n", N_runs);
		} else if (k==11) {
			sscanf (line,"%s %s %*s %d",str,str,&N_episodes);
			//~ printf("        N episodes = %d\n", N_episodes);
		} else if (k==12) {
			sscanf (line,"%s %*s %lf",str,&time_single_episode);
			//~ printf("        T = %11.8f\n", time_single_episode);
		} else if (k==13) {
			sscanf (line,"%s %*s %lf",str,&max_phase_duration);
			//~ printf("        PT = %11.8f\n", max_phase_duration);
		} else if (k==14) {
			sscanf (line,"%s %*s %lf",str,&reward);
			//~ printf("        reward = %11.8f\n", reward);
		} else if (k==15) {
			sscanf (line,"%s %*s %lf",str,&gammma);
			//~ printf("        gamma = %11.8f\n", gammma);
		} else if (k==16) {
			sscanf (line,"%s %*s %lf",str,&eta);
			//~ printf("        eta = %11.8f\n", eta);
		} else if (k==17) {
			sscanf (line,"%s %s %s %*s %d",str,str,str,&const_initial_policy);
			//~ printf("        const_initial_policy = %d\n", const_initial_policy);
		}
		
		if (const_initial_policy == 1) {
			if (k==18) {
				sscanf (line,"%s %*s %lf",str,&pBP);
				//~ printf("        pBP = %11.8f\n", pBP);
			} else if (k==19) {
				sscanf (line,"%s %*s %lf",str,&pABP);
				//~ printf("        pABP = %11.8f\n", pABP);
			}
		}
	}
	fclose(fp);
	if (line) {
		free(line);
	}
	
	// ... it follows
	D = L*L/(4*tau);		// diffusion coefficient for the passive particle (kept the same also for the ABP phase)
	v = Pe * L / tau;		// self-propulsion velocity (it must be such that typically the particle do not meet the target)
	D_theta = v/ell;		// rotational diffusion coefficient
	Nt_single_episode = int(time_single_episode/dt);
	Nt_max_phase_duration = int(max_phase_duration/dt);
	
	N_runs0 = k_bunch*N_runs;
	
	// check and print parameters
	printf("typical step length due to diffusion =  %f L \n",sqrt(4*D*dt));
	printf("typical step length due to self-propulsion =  %f L \n",v*dt);
	printf("bunch %d\n",k_bunch);
	printf("running from episode %d to episode %d\n",N_episodes0,N_episodes0+N_episodes-1);
}

void array_initialization(){
	target_times = dmatrix(0,N_runs-1,0,N_episodes-1);			// not more than 100 target found in a episode. Otherwise this number has to change both in this file and in the pyx file
	H = dmatrix(0,Ns-1,0,Na-1);				// h-values (the number of states should match with Nt_max_phase_duration)
	H0 = dmatrix(0,Ns-1,0,Na-1);			// initial h-values (the number of states should match with Nt_max_phase_duration)
	G = dmatrix(0,Ns-1,0,Na-1);				// Glow matrix
	Pvalues = dmatrix(0,N_runs-1,0,Ns-1);
	Svalues = dmatrix(0,N_runs-1,0,Ns-1);
	avg_Pvalues = dvector(0,Ns-1);
	avg_Pvalues_episode = dmatrix(0,N_episodes-1,0,Ns-1);
	avg_H0values_episode = dmatrix(0,N_episodes-1,0,Ns-1);
	avg_H1values_episode = dmatrix(0,N_episodes-1,0,Ns-1);
	avg_state_action_values = dmatrix(0,Ns-1,0,Na-1);
	avg_glow = dmatrix(0,Ns-1,0,Na-1);
	avg_target_times = dvector(0,N_episodes-1);
	changes_in_G = imatrix(0,Ns-1,0,Na-1);
	vBP = dvector(0,Nt_max_phase_duration-1);
	vABP = dvector(0,Nt_max_phase_duration-1);
	
	// initializations
	if (const_initial_policy == 1){
		for (int s = 0; s<Ns/2; s++){
			vBP[s]=pBP;
			vABP[s]=pABP;
		}
	} else {
		if (N_episodes0 == 0) {
			for (int s = 0; s<Ns/2; s++){
				//~ vBP[s]= pBP + (1.0-pBP) * (s+1)*1./(Ns/2);
				//~ vABP[s]= pABP + (1.0-pABP) * (s+1)*1./(Ns/2);
				if (s<500){
					vBP[s]= 0.01;
					vABP[s]= 0.01;
				}else{
					vBP[s]= 0.99;
					vABP[s]= 0.99;
				}
			}
			print_initial_policy();
		} else {
			// read initial policy
		}
	}
	for (int s = 0; s<Ns/2; s++){
		H0[s][0]=1.0-vBP[s];
		H0[s][1]=vBP[s];
	}
	for (int s = Ns/2; s<Ns; s++){
		H0[s][0]=1.0-vABP[s-Ns/2];
		H0[s][1]=vABP[s-Ns/2];
	}
	
	for (int s = 0; s<Ns; s++){
		avg_Pvalues[s]=0.0;
		for (int a=0; a<Na; a++){
			avg_state_action_values[s][a]=0.0;
			avg_glow[s][a]=0.0;
		}
	}
	for (int episode=0; episode<N_episodes; episode++){
		avg_target_times[episode] = 0.;
		for (int s = 0; s<Ns; s++){
			avg_Pvalues_episode[episode][s]=0.0;
			avg_H0values_episode[episode][s]=0.0;
			avg_H1values_episode[episode][s]=0.0;
		}
	}
	for (int run=0; run<N_runs; run++){
		for (int episode=0; episode<N_episodes; episode++){
			target_times[run][episode] = 0.;
		}
	}
	
	// read the H learned at the previous step
	if (N_episodes0 != 0) {// if it is a restart
		H_restart = d3tensor(0,N_runs,0,Ns-1,0,Na-1);
		read_H_restart();
	}
}

void print_initial_policy() {
	std::stringstream sstr;
	sstr << "InitialPolicy.txt";
	const std::string tmp = sstr.str();
	const char* cstr = tmp.c_str();
	FILE * out = fopen(cstr,"w");
	for (int s = 0; s<Ns/2; s++){
		fprintf(out,"%5d   %14.8f  %14.8f \n" , s,vBP[s],vABP[s]);
	}
	fclose(out);
}

void print_averages () {
	std::stringstream sstr;
	sstr << "AvgHvalues_bunch" << k_bunch << "_episode" <<  N_episodes0+N_episodes  << ".dat";
	const std::string tmp = sstr.str();
	const char* cstr = tmp.c_str();
	FILE * out = fopen(cstr,"w");
	for (int s = 0; s<Ns/2; s++){
		fprintf(out,"%5d %2d  %14.8f \n" , s,0,avg_state_action_values[s][0]);
		fprintf(out,"%5d %2d  %14.8f \n" , s,1,avg_state_action_values[s][1]);
		fprintf(out,"%5d %2d  %14.8f \n" , s+Ns/2,0,avg_state_action_values[s+Ns/2][0]);
		fprintf(out,"%5d %2d  %14.8f \n" , s+Ns/2,1,avg_state_action_values[s+Ns/2][1]);
	}
	fclose(out);

	std::stringstream sstr2;
	sstr2 << "AvgGvalues_bunch" << k_bunch << "_episode" <<  N_episodes0+N_episodes  << ".dat";
	const std::string tmp2 = sstr2.str();
	const char* cstr2 = tmp2.c_str();
	FILE * out2 = fopen(cstr2,"w");
	for (int s = 0; s<Ns/2; s++){
		fprintf(out2,"%5d %2d  %12.2f \n" , s,0,avg_glow[s][0]);
		fprintf(out2,"%5d %2d  %12.2f \n" , s,1,avg_glow[s][1]);
		fprintf(out2,"%5d %2d  %12.2f \n" , s+Ns/2,0,avg_glow[s+Ns/2][0]);
		fprintf(out2,"%5d %2d  %12.2f \n" , s+Ns/2,1,avg_glow[s+Ns/2][1]);
	}
	fclose(out2);

	std::stringstream sstr3;
	sstr3 << "AvgPvalues_bunch" << k_bunch << "_episode" <<  N_episodes0+N_episodes  << ".dat";
	const std::string tmp3 = sstr3.str();
	const char* cstr3 = tmp3.c_str();
	FILE * out3 = fopen(cstr3,"w");
	for (int s = 0; s<Ns/2; s++){
		fprintf(out3,"%5d  %10.8f %10.8f \n" , s,avg_Pvalues[s],avg_Pvalues[s+Ns/2] );
	}
	fclose(out3);
	
	
	std::stringstream sstr4;
	sstr4 << "AvgTargetTimes_bunch" << k_bunch << ".dat";
	const std::string tmp4 = sstr4.str();
	const char* cstr4 = tmp4.c_str();
	FILE * out4 = fopen(cstr4,"a");
	for (int ep = 0; ep<N_episodes; ep++){
		fprintf(out4,"%6d  %10.6f\n" , N_episodes0+ep,avg_target_times[ep]);
	}
	fclose(out4);
	
	std::stringstream sstr5;
	sstr5 << "AvgPvaluesEpisode_bunch" << k_bunch << ".dat";
	const std::string tmp5 = sstr5.str();
	const char* cstr5 = tmp5.c_str();
	FILE * out5 = fopen(cstr5,"a");
	for (int ep = 0; ep<N_episodes; ep++){
		for (int s = 0; s<Ns/2; s++){
			fprintf(out5,"%6d %5d  %10.8f %10.8f \n" , N_episodes0+ep,s,avg_Pvalues_episode[ep][s],avg_Pvalues_episode[ep][s+Ns/2] );
		}
	}
	fclose(out5);
	
	std::stringstream sstr6;
	sstr6 << "AvgHvaluesEpisode_bunch" << k_bunch << ".dat";
	const std::string tmp6 = sstr6.str();
	const char* cstr6 = tmp6.c_str();
	FILE * out6 = fopen(cstr6,"a");
	for (int ep = 0; ep<N_episodes; ep++){
		for (int s = 0; s<Ns/2; s++){
			fprintf(out6,"%6d %5d  %10.8f %10.8f  %10.8f %10.8f \n" , N_episodes0+ep,s,avg_H0values_episode[ep][s],avg_H1values_episode[ep][s],avg_H0values_episode[ep][s+Ns/2],avg_H1values_episode[ep][s+Ns/2] );
		}
	}
	fclose(out6);
}

void print_data () {
	std::stringstream sstr2;
	sstr2 << "Pvalues_bunch" << k_bunch << "_episode" <<  N_episodes0+N_episodes  << ".dat";
	const std::string tmp2 = sstr2.str();
	const char* cstr2 = tmp2.c_str();
	FILE * out2 = fopen(cstr2,"w");
	for (int run=0; run<N_runs; run++){
		for (int s = 0; s<Ns/2; s++){
			fprintf(out2,"%5d %5d  %10.8f %10.8f  %10.4f %10.4f\n" , N_runs0+run,s,Pvalues[run][s],Pvalues[run][s+Ns/2],Svalues[run][s],Svalues[run][s+Ns/2]);
		}
	}
	fclose(out2);
	
	std::stringstream sstr3;
	sstr3 << "TargetTimes_bunch" << k_bunch << ".dat";
	const std::string tmp3 = sstr3.str();
	const char* cstr3 = tmp3.c_str();
	FILE * out3 = fopen(cstr3,"a");
	for (int run=0; run<N_runs; run++){
		for (int episode=0; episode<N_episodes; episode++){
			fprintf(out3,"%5d %6d %10.4f\n" , N_runs0+run,N_episodes0+episode,target_times[run][episode]);
		}
	}
	fclose(out3);
}

void read_H_restart () {
	std::stringstream sstr2;
	sstr2 << "Pvalues_bunch" << k_bunch << "_episode" <<  N_episodes0  << ".dat";
	const std::string tmp2 = sstr2.str();
	const char* cstr2 = tmp2.c_str();
	FILE* fp = fopen(cstr2, "r");
	if (fp == NULL) {
		exit(EXIT_FAILURE);
	}
	char str [20];
	char* line = NULL;
	size_t len = 0;
	int k = -1;
	
	int run;
	int rrun;
	int ss;
	double probBP;
	double probABP;
	double HstayBP;
	double HstayABP;
	
	while ((getline(&line, &len, fp)) != -1) {
		// using printf() in all tests for consistency
		k++;
		sscanf (line,"%d %d %lf %lf %lf %lf",&rrun,&ss,&probBP,&probABP,&HstayBP,&HstayABP);
		run = rrun % N_runs;
		H_restart[run][ss][0] = HstayBP;
		H_restart[run][ss+Ns/2][0] = HstayABP;
		H_restart[run][ss][1] = HstayBP*probBP/(1.0-probBP);
		H_restart[run][ss+Ns/2][1] = HstayABP*probABP/(1.0-probABP);
		//~ if (ss==0) {
			//~ printf("%d %s", k,line);
			//~ printf("%d %d   %10.8f %10.8f   %10.4f %10.4f\n",run,ss,probBP,probABP,HstayBP,HstayABP);
			//~ printf("%10.4f %10.4f   %10.4f %10.4f\n\n",H_restart[run][ss][0],H_restart[run][ss][1],H_restart[run][ss+Ns/2][0],H_restart[run][ss+Ns/2][1]);
		//~ }
	}
	fclose(fp);
	if (line) {
		free(line);
	}
}

void ProjectiveSimulationOptimized(int run){
	// the state s=(phi,omega) include the phase phi=BP or ABP and the duration omega of the current phase
	// the action a is stay in current phase (a=0) or switch phase(a=1)
	int phi;							// phase
	int omega;							// current-phase duration
	double x,y;							// particle position
	double theta;						// self-propulsion direction
	double x_target,y_target;			// target position
	
	double sigma = sqrt(2*D*dt);
	double sigma_theta = sqrt(2*D_theta*dt);
	double vdt = v*dt;
	
	double dx,dy,phiold;
	double p;							// probability of switching given that we are in a given state
	int n_targets,s,a,t0,ks;
	double avg_target_time_ep;
	double factor1, factor2;

	// initial particle position
	x = (rand()/((double) RAND_MAX)) * L;
	y = (rand()/((double) RAND_MAX)) * L;
	theta = 2*PI*(rand()/((double) RAND_MAX));
	
	// initial h-values matrix
	if (N_episodes0 ==0 ) {
		for (int ss = 0; ss<Ns; ss++){
			H[ss][0]=H0[ss][0];
			H[ss][1]=H0[ss][1];
		}
	} else {
		for (int ss = 0; ss<Ns; ss++){
			H[ss][0]=H_restart[run][ss][0];
			H[ss][1]=H_restart[run][ss][1];
		}
	}

	// loop over episodes
	for (int episode=0; episode<N_episodes; episode++){
		// initialization for the episode
		avg_target_time_ep = 0.0;
		n_targets = 0;
		phi = 0;
		phiold = 0;
		omega=0;
		t0=-1;
		ks=0;
		
		// initialize new target position
		x_target = (rand()/((double) RAND_MAX)) * L;
		y_target = (rand()/((double) RAND_MAX)) * L;
		dx = x-x_target;
		if (dx > L/2) dx -= L;
		if (dx < -L/2) dx += L;
		dy = y-y_target;
		if (dy > L/2) dy -= L;
		if (dy < -L/2) dy += L;
		while (sqrt(dx*dx+dy*dy) <= R_target){
			x_target = (rand()/((double) RAND_MAX)) * L;
			y_target = (rand()/((double) RAND_MAX)) * L;
			dx = x-x_target;
			if (dx > L/2) dx -= L;
			if (dx < -L/2) dx += L;
			dy = y-y_target;
			if (dy > L/2) dy -= L;
			if (dy < -L/2) dy += L;
		}
		
		// initialize to zero the glow-matrix
		for (int ss = 0; ss<Ns; ss++){
			G[ss][0]=0.0;
			G[ss][1]=0.0;
			changes_in_G[ss][0]=0;
			changes_in_G[ss][1]=0;
		}
		
		// run dynamics
		for (int t=0; t<Nt_single_episode; t++){
			a = 0;
			if (omega<Nt_max_phase_duration){
				s = phi*Nt_max_phase_duration + omega;
				H2[0] = v_factor1[t-t0-1]*H[s][0] + v_factor2[t-t0-1]*H0[s][0];
				H2[1] = v_factor1[t-t0-1]*H[s][1] + v_factor2[t-t0-1]*H0[s][1];
				p = H2[1]/(H2[1]+H2[0]);
			} else{
				p = 1.00001;
			}
			
			// switch with probability pi
			if ((rand()/((double) RAND_MAX)) <= p){
				phi = (phi+1) % Nphi;
				omega = 0;
				a = 1;
			} else{
				omega += 1;
			}
				
			// update position
			double u1 = rand ()/(( double ) RAND_MAX );
			double u2 = rand ()/(( double ) RAND_MAX );
			double z1 = sqrt ( -2.* log ( u1 )) * cos (2.* PI * u2 );
			if (z1<-6660.0 || z1>6660.0){
				u1 = rand ()/(( double ) RAND_MAX );
				u2 = rand ()/(( double ) RAND_MAX );
				z1 = sqrt ( -2.* log ( u1 )) * cos (2.* PI * u2 );
			}
			x += phiold*vdt*cos(theta) + sigma * z1;
			if (x>L) x -= L;
			if (x<0.) x += L;
			u1 = rand ()/(( double ) RAND_MAX );
			u2 = rand ()/(( double ) RAND_MAX );
			z1 = sqrt ( -2.* log ( u1 )) * cos (2.* PI * u2 );
			if (z1<-6660.0 || z1>6660.0){
				u1 = rand ()/(( double ) RAND_MAX );
				u2 = rand ()/(( double ) RAND_MAX );
				z1 = sqrt ( -2.* log ( u1 )) * cos (2.* PI * u2 );
			}
			y += phiold*vdt*sin(theta) + sigma * z1;
			if (y>L) y -= L;
			if (y<0.) y += L;
			u1 = rand ()/(( double ) RAND_MAX );
			u2 = rand ()/(( double ) RAND_MAX );
			z1 = sqrt ( -2.* log ( u1 )) * cos (2.* PI * u2 );
			if (z1<-6660.0 || z1>6660.0){
				u1 = rand ()/(( double ) RAND_MAX );
				u2 = rand ()/(( double ) RAND_MAX );
				z1 = sqrt ( -2.* log ( u1 )) * cos (2.* PI * u2 );
			}
			theta += sigma_theta * z1;
			if (omega==0 and t>0) theta = 2*PI*(rand()/((double) RAND_MAX));
			phiold = phi;
			
			// update matrices that I need to update Glow matrix when I find the target
			if (omega < Nt_max_phase_duration){
				changes_in_G[s][a]=1;
				visited_states[ks]=s;
				visited_actions[ks]=a;
				ks+=1;
			}
			
			// check if we have reached the target (only possible in passive case)
			if (phi == 0){
				dx = x-x_target;
				if (dx > L/2) dx -= L;
				if (dx < -L/2) dx += L;
				dy = y-y_target;
				if (dy > L/2) dy -= L;
				if (dy < -L/2) dy += L;
				if (sqrt(dx*dx+dy*dy) <= R_target){
					// update number of targets within the episode
					avg_target_time_ep += (t-t0)*dt;
					n_targets += 1;
					// initialize new target position
					x_target = (rand()/((double) RAND_MAX)) * L;
					y_target = (rand()/((double) RAND_MAX)) * L;
					dx = x-x_target;
					if (dx > L/2) dx -= L;
					if (dx < -L/2) dx += L;
					dy = y-y_target;
					if (dy > L/2) dy -= L;
					if (dy < -L/2) dy += L;
					while (sqrt(dx*dx+dy*dy) <= R_target){
						x_target = (rand()/((double) RAND_MAX)) * L;
						y_target = (rand()/((double) RAND_MAX)) * L;
						dx = x-x_target;
						if (dx > L/2) dx -= L;
						if (dx < -L/2) dx += L;
						dy = y-y_target;
						if (dy > L/2) dy -= L;
						if (dy < -L/2) dy += L;
					}
					
					// update Glow matrix using visited_state-action_pairs and changes_in_G (up to previous step)
					factor1 = pow((1.0-eta),(t-t0-1));
					for (int i=0; i<Ns; i++){
						for (int j=0; j<Na; j++){
							if (changes_in_G[i][j]==1){
								for (int k=0; k<ks-1; k++){
									if (i == visited_states[k] && j == visited_actions[k]) G[i][j] += 1.0;
									G[i][j] *= (1.0-eta);
								}
								changes_in_G[i][j]=0;
							} else{
								G[i][j]=factor1*G[i][j];
							}
						}
					}
					// update Glow matrix for the last visited state-action pair
					G[visited_states[ks-1]][visited_actions[ks-1]] += 1.0;
					
					// update h-values with rewards if target is found
					for (int i=0; i<Ns; i++){
						H[i][0] = v_factor1[t-t0]*H[i][0] + v_factor2[t-t0]*H0[i][0] + reward*G[i][0];
						H[i][1] = v_factor1[t-t0]*H[i][1] + v_factor2[t-t0]*H0[i][1] + reward*G[i][1];
					}
					
					// damp glow matrix after finding the target
					for (int i=0; i<Ns; i++){
						G[i][0] = (1.0-eta)*G[i][0];
						G[i][1] = (1.0-eta)*G[i][1];
					}
					
					// update time at which the target is found
					t0=t;
					ks=0;
					
				}	
			}
			
			// update H and G if more that 100000 have been visited without meeting a target
			if (t-t0>=100000){
				for (int i=0; i<Ns; i++){
					H[i][0] = v_factor1[t-t0]*H[i][0] + v_factor2[t-t0]*H0[i][0];
					H[i][1] = v_factor1[t-t0]*H[i][1] + v_factor2[t-t0]*H0[i][1];
					factor1 = pow((1.0-eta),(t-t0));
				}
				for (int i=0; i<Ns; i++){
					for (int j=0; j<Na; j++){
						if (changes_in_G[i][j]==1){
							for (int k=0; k<ks; k++){
								if (i == visited_states[k] && j == visited_actions[k]) G[i][j] += 1.0;
								G[i][j] *= (1.0-eta);
							}
							changes_in_G[i][j]=0;
						} else{
							G[i][j]=factor1*G[i][j];
						}
					}
				}
				t0=t;
				ks=0;
			}
		}
		
		// last update of the h-values
		int t2 = Nt_single_episode-1;
		if (t2>t0){
			for (int i=0; i<Ns; i++){
				H[i][0] = v_factor1[t2-t0]*H[i][0] + v_factor2[t2-t0]*H0[i][0];
				H[i][1] = v_factor1[t2-t0]*H[i][1] + v_factor2[t2-t0]*H0[i][1];
			}
		}
	
		// compute average duration of targets during the episode
		if (n_targets > 0){
			target_times[run][episode]=avg_target_time_ep/n_targets;
		} else {
			target_times[run][episode] = time_single_episode;
		}
		// update average Probability values at the end of the episode
		for (int s = 0; s<Ns; s++){
			avg_Pvalues_episode[episode][s] += H[s][1]/(H[s][0]+H[s][1]);
			avg_H0values_episode[episode][s] += H[s][0];
			avg_H1values_episode[episode][s] += H[s][1];
		}
	}
}

