#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <pangolin/pangolin.h>

#include <iostream>
#include <cstdio>
#include <sstream>
#include <algorithm>
#include <iterator>
using namespace std;
using namespace cv;
using namespace Eigen;

// int WIDTH = 1226;
// int HEIGHT = 370;
int WIDTH = 1241;
int HEIGHT = 376;
int nImgs = 19;
int MAX_CNT = 300;
int MIN_DIST = 30;
double fx = 7.070912e+02;
double fy = 7.070912e+02;
double cx = 6.018873e+02;
double cy = 1.831104e+02;

struct SFMFeature
{
    bool state;
    int id;
    vector<pair<int,Vector2d>> observation;
    double position[3];
    double depth;
    SFMFeature()
    {
    	state = false;
    	id = -1;
    	depth = -1;
    }
};

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < WIDTH - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < HEIGHT - BORDER_SIZE;
}

void setMask(Mat& mask,vector<Point2f>& pts)
{
	for(auto& pt:pts)
	{
		// if(mask.at<uchar>(pt) == 255)
		// {
		circle(mask, pt, MIN_DIST, 0, -1);
		//}
	}
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}
void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

int nCorres(int l, int r, vector<SFMFeature>& sfm)
{
	int c = 0;
	for(auto& s:sfm)
	{
		int ll = s.observation[0].first;
		int rr = s.observation[s.observation.size() - 1].first;
		if(ll <= l && rr >= r)
			c++;
	}
	return c;
}

Point2d pixel2cam ( const Point2d& p )
{
    return Point2d
           (
               ( p.x - cx ) / fx,
               ( p.y - cy ) / fy
           );
}

vector<pair<Vector2d, Vector2d>> getCorrespondings(int l, int r, vector<SFMFeature>& sfm)
{
	vector<pair<Vector2d, Vector2d>> corres;
	for(auto& s:sfm)
	{
		int ll = s.observation[0].first;
		int rr = s.observation[s.observation.size() - 1].first;
		if(ll <= l && rr >= r)
		{
			corres.push_back(make_pair(s.observation[l-ll].second, s.observation[r-ll].second));
		}
	}
	return corres;
}

bool solveRelativeRT(const vector<pair<Vector2d, Vector2d>> &corres, Matrix3d &Rotation, Vector3d &Translation)
{
	if (corres.size() >= 15)
    {
        vector<cv::Point2f> ll, rr;
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1)));
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1)));
        }
        cv::Mat mask;
        cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat rot, trans;
        int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);
        //cout << "inlier_cnt " << inlier_cnt << endl;

        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        for (int i = 0; i < 3; i++)
        {   
            T(i) = trans.at<double>(i, 0);
            for (int j = 0; j < 3; j++)
                R(i, j) = rot.at<double>(i, j);
        }

        Rotation = R.transpose();
        Translation = -R.transpose() * T;
        if(inlier_cnt > 12)
            return true;
        else
            return false;
    }
    return false;
}

void triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero();
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
	point_3d(0) = triangulated_point(0) / triangulated_point(3);
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}

void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
						  int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
						  vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1);
	for (int j = 0; j < sfm_f.size(); j++)
	{
		if (sfm_f[j].state == true)
			continue;
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == frame0)
			{
				point0 = sfm_f[j].observation[k].second;
				has_0 = true;
			}
			if (sfm_f[j].observation[k].first == frame1)
			{
				point1 = sfm_f[j].observation[k].second;
				has_1 = true;
			}
		}
		if (has_0 && has_1)
		{
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d);
			sfm_f[j].state = true;
			sfm_f[j].position[0] = point_3d(0);
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}

bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector;
	vector<cv::Point3f> pts_3_vector;
	for (int j = 0; j < sfm_f.size(); j++)
	{
		if (sfm_f[j].state != true)
			continue;
		Vector2d point2d;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++)
		{
			if (sfm_f[j].observation[k].first == i)
			{
				Vector2d img_pts = sfm_f[j].observation[k].second;
				cv::Point2f pts_2(img_pts(0), img_pts(1));
				pts_2_vector.push_back(pts_2);
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]);
				pts_3_vector.push_back(pts_3);
				break;
			}
		}
	}
	if (int(pts_2_vector.size()) < 15)
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10)
			return false;
	}
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r);
	cv::Rodrigues(tmp_r, rvec);
	cv::eigen2cv(P_initial, t);
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
	bool pnp_succ;
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
	if(!pnp_succ)
	{
		return false;
	}
	cv::Rodrigues(rvec, r);
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp;
	cv::cv2eigen(r, R_pnp);
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp);
	R_initial = R_pnp;
	P_initial = T_pnp;
	return true;

}

void Draw(Eigen::Matrix<double, 3, 4> poses[], vector<SFMFeature>& sfm) 
{
    if (sfm.empty()) {
        cerr << "parameter is empty!" << endl;
        return;
    }
    float fx = 277.34;
    float fy = 291.402;
    float cx = 312.234;
    float cy = 239.777;
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        // draw poses
        float sz = 0.1;
        int width = 640, height = 480;
        // for (auto &Tcw: poses) {
        for(int i = 0; i < nImgs; i++)
        {
        	glPushMatrix();
	        //Sophus::Matrix4f m = Tcw.inverse().matrix().cast<float>();
	        Matrix4d current_pos = Matrix4d::Identity();
	        current_pos.block<3,4>(0,0) = poses[i];
	        current_pos = current_pos.inverse();
	        glMultMatrixd((GLdouble *) current_pos.data());
	        glColor3f(1, i/12.0, 0);
	        glLineWidth(2);
	        glBegin(GL_LINES);
	        glVertex3f(0, 0, 0);
	        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
	        glVertex3f(0, 0, 0);
	        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	        glVertex3f(0, 0, 0);
	        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	        glVertex3f(0, 0, 0);
	        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
	        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
	        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	        glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	        glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
	        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
	        glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
	        glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
	        glEnd();
	        glPopMatrix();
        }
        
        // }

        // points
        glPointSize(2);
        glBegin(GL_POINTS);
        for (size_t i = 0; i < sfm.size(); i++) 
        {
        	if(sfm[i].state == true)
        	{
        		glColor3f(0, 1, 0.7);
            	glVertex3d(sfm[i].position[0], sfm[i].position[1], sfm[i].position[2]);
        	}
            
        }
        glEnd();

        // glLineWidth(2);
        // if(trj.size() > 1)
        // {
        //     for(int i = 0;i < trj.size() - 1;i++)
        //     {
        //         glColor3f(0, 1, 0);
        //         glBegin(GL_LINES);
        //         auto p1 = trj[i], p2 = trj[i + 1];
        //         glVertex3d(p1[0], p1[1], p1[2]);
        //         glVertex3d(p2[0], p2[1], p2[2]);
        //         glEnd();
        //     }
        // }

        // m_img.lock();
        // imshow("img",img);
        // waitKey(5);
        // // imageTexture.Upload(img.data,GL_RGB,GL_UNSIGNED_BYTE);
        // // d_image.Activate();
        // // glColor3f(1.0,1.0,1.0);
        // // imageTexture.RenderToViewport();
        // m_img.unlock();
        pangolin::FinishFrame();
        
        //usleep(5000);   // sleep 5 ms
    }
}

bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l, vector<SFMFeature>& sfm)
{
    // find previous frame which contains enough correspondance and parallex with newest frame
    for (int i = 0; i < nImgs - 1; i++)
    {
        vector<pair<Vector2d, Vector2d>> corres;
        corres = getCorrespondings(i, nImgs-1, sfm);
        if (corres.size() > 30)
        {
            double sum_parallax = 0;
            double average_parallax;
            for (int j = 0; j < int(corres.size()); j++)
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1));
                Vector2d pts_1(corres[j].second(0), corres[j].second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax = sum_parallax + parallax;

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size());
            if(average_parallax * 460 > 30 && solveRelativeRT(corres, relative_R, relative_T))//视差大于阈值且能够成功计算相对变换，令l为i
            {
                l = i;
                printf("average_parallax %f choose l %d and newest frame to triangulate the whole structure\n", average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

int main()
{
	Mat cur_frame,frw_frame;
	vector<Point2f> cur_pts,frw_pts;
	vector<uchar> status;
    vector<float> err;
    Mat msk0(HEIGHT,WIDTH, CV_8UC1, cv::Scalar(255));
 	vector<SFMFeature> sfm;
 	vector<int> ids;
 	unsigned long id = 0;

	for(int i = 0;i<nImgs;i++)
	{

		Mat mask = msk0.clone();
		stringstream sstr;
		//sstr <<"../10_930_940/"<<"000"<<i+930<<".png";
		sstr <<"../19_460_518/"<<"000"<<i+468<<".png";
		Mat frw_frame = imread(sstr.str(),0);
		//GaussianBlur(frw_frame,frw_frame,Size(5,5),0);
		if(i == 0)
		{
			goodFeaturesToTrack(frw_frame,cur_pts,MAX_CNT,0.01,MIN_DIST);
			for(auto &c:cur_pts)
				ids.push_back(-1);
			//frw_pts = cur_pts;
			cur_frame = frw_frame;
			// for(auto& pt:cur_pts)
			// 	circle(img, pt, 2, cv::Scalar(0, 0, 255), 2);
			continue;
		
		}

		calcOpticalFlowPyrLK(cur_frame, frw_frame, cur_pts, frw_pts, status, err, cv::Size(30, 30), 3);
		
		for(int j = 0;j < cur_pts.size();j++)
		{
			if(ids[j] < 0 && status[j])
				ids[j] = id++;
		}
		//cout<<ids.size()<<" "<<cur_pts.size()<<endl;

		reduceVector(ids, status);
		reduceVector(frw_pts, status);
		reduceVector(cur_pts, status);
		vector<uchar> status2;
		findFundamentalMat(cur_pts, frw_pts, cv::FM_RANSAC, 1.0, 0.99, status2);
		reduceVector(frw_pts,status2);
		reduceVector(cur_pts,status2);
		reduceVector(ids, status2);

		for(int j = 0; j < ids.size(); j++)
		{
			vector<SFMFeature>::iterator it;
			it = find_if(sfm.begin(), sfm.end(), [&](SFMFeature s){return s.id == ids[j];});
			if(it == sfm.end())
			{
				//cout<<"create new sfm feature."<<endl;
				SFMFeature sfm_feat;
				sfm_feat.id = ids[j];
				Point2d p0,p1;
				p0 = pixel2cam(Point2d(cur_pts[j].x, cur_pts[j].y));
				p1 = pixel2cam(Point2d(frw_pts[j].x, frw_pts[j].y));
				sfm_feat.observation.push_back(make_pair(i-1, Vector2d(p0.x, p0.y)));
				sfm_feat.observation.push_back(make_pair(i, Vector2d(p1.x, p1.y)));
				sfm.push_back(sfm_feat);
			}
			else
			{
				//cout<<"append measurement."<<endl;
				Point2d p1 = pixel2cam(Point2d(frw_pts[j].x, frw_pts[j].y));
				it->observation.push_back(make_pair(i, Vector2d(p1.x, p1.y)));
			}
		}
		setMask(mask,frw_pts);
		//imshow("mask",mask);
		vector<Point2f> newFeat;

		goodFeaturesToTrack(frw_frame,newFeat,MAX_CNT,0.01,MIN_DIST,mask);

		Mat disp;
		cvtColor(frw_frame,disp,CV_GRAY2BGR);
		
		// for(auto& pt:frw_pts)
		// {
		// 	circle(disp, pt, 2, cv::Scalar(0, 0, 255), 2);
		// }
		//Mat msk = mask.clone();
		//cvtColor(msk,msk,CV_GRAY2BGR);
		for(int j = 0;j < cur_pts.size();j++)
		{
			if(inBorder(frw_pts[j]) && msk0.at<uchar>(frw_pts[j].y,frw_pts[j].x) == 255)
			{
				circle(disp, frw_pts[j], 2, cv::Scalar(0, 0, 255), 2);
				line(disp,cur_pts[j],frw_pts[j],Scalar(0,255,255));
				//circle(msk, frw_pts[j], 2, cv::Scalar(0, 0, 255), 2);
			}
			else
			{
				//circle(msk, frw_pts[j], 2, cv::Scalar(255, 0, 0), 2);
				circle(disp, frw_pts[j], 2, cv::Scalar(255, 0, 0), 2);
			}
			// stringstream ss;
			// ss<<err[j];
			

		}
		// for(auto& pt:newFeat)
		// {
		// 	//circle(msk, pt, 2, cv::Scalar(0, 255, 0), 2);
		// 	//circle(disp, pt, 2, cv::Scalar(0, 255, 0), 2);
		// }
		
		// for(auto &s:sfm)
		// {
		// 	cout<<"point id: "<<s.id<<endl;
		// 	for(auto& ob:s.observation)
		// 		cout<<ob.first<<" "<<ob.second.transpose()<<"   ";
		// 	cout<<endl;
		// }

		cur_frame = frw_frame;
		cur_pts = frw_pts;
		for(auto& pt:newFeat)
		{
			cur_pts.push_back(pt);
			ids.push_back(-1);
		}
		frw_pts.clear();
		//imshow("msk",msk);
		imshow("frame",disp);
		
		waitKey(200);
	}
	// for(int i = 0; i < 10; i++)
	// 	cout<<nCorres(i, 10, sfm)<<endl;
	// vector<Matrix3d> Rs(nImgs);
	// vector<Vector3d> Ps(nImgs);

	int l = -1;

	// vector<pair<Vector2d,Vector2d>> corres = getCorrespondings(l, nImgs - 1, sfm);
	Matrix3d R;
	Vector3d P;
	// solveRelativeRT(corres, R, P);//投影矩阵·

	if(!relativePose(R, P, l, sfm))
	{
		cout<<"failed to find appropriate reference frame."<<endl;
		return -1;
	}

	// Rs[nImgs-1] = R;
	// Ps[nImgs-1] = P;
	// Rs[l] = Matrix3d::Identity();
	// Ps[l] = Vector3d(0,0,0);
	Matrix3d c_Rotation[nImgs];
	Vector3d c_Translation[nImgs];
	c_Rotation[l] = Matrix3d::Identity();
	c_Translation[l] = Vector3d(0,0,0);
	c_Rotation[nImgs-1] = R.transpose();
	c_Translation[nImgs-1] = - R.transpose()*P;

	Eigen::Matrix<double, 3, 4> Pose[nImgs];
	Pose[l].block<3,3>(0,0) = c_Rotation[l];
	Pose[l].block<3,1>(0,3) = c_Translation[l];
	Pose[nImgs-1].block<3,3>(0,0) = c_Rotation[nImgs-1];
	Pose[nImgs-1].block<3,1>(0,3) = c_Translation[nImgs-1];


	//1: trangulate between l ----- frame_num - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1; 
	for (int i = l; i < nImgs - 1 ; i++)
	{
		//solve pnp
		if (i > l)
		{
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm))
			{
				cout<<"failed to solvePnP."<<endl;
				return -1;
			}
			//cout<<"solvePnP success"<<endl;
			c_Rotation[i] = R_initial;
			c_Translation[i] = P_initial;
			
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		// triangulate point based on the solve pnp result
		triangulateTwoFrames(i, Pose[i], nImgs - 1, Pose[nImgs - 1], sfm);
	}

	for (int i = l + 1; i < nImgs - 1; i++)
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm);

	for (int i = l - 1; i >= 0; i--)
	{
		//solve pnp
		Matrix3d R_initial = c_Rotation[i + 1];
		Vector3d P_initial = c_Translation[i + 1];
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm))
		{
			cout<<"failed to solvePnP."<<endl;
			return -1;
		}
		c_Rotation[i] = R_initial;
		c_Translation[i] = P_initial;
		//c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm);
	}

	for (int j = 0; j < sfm.size(); j++)
	{
		if (sfm[j].state == true)
			continue;
		if ((int)sfm[j].observation.size() >= 2)
		{
			Vector2d point0, point1;
			int frame_0 = sfm[j].observation[0].first;
			point0 = sfm[j].observation[0].second;
			int frame_1 = sfm[j].observation.back().first;
			point1 = sfm[j].observation.back().second;
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d);
			sfm[j].state = true;
			sfm[j].position[0] = point_3d(0);
			sfm[j].position[1] = point_3d(1);
			sfm[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}		
	}

	for(int i = 0; i < nImgs; i++)
	{
		cout<<"Pose "<<i<<endl;
		cout<<Pose[i]<<endl<<endl;
	}
		
		// cout<<Pose[l]<<endl;
		// cout<<Pose[nImgs-1]<<endl;
	// cout<<solveRelativeRT(corres, R, P)<<endl;
	// cout<<R<<endl;
	// cout<<P.transpose()<<endl;
	Draw(Pose, sfm);
	return 0;
}